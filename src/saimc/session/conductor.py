"""The conductor: one model call per turn, and the tools it asks for.

`tools.py` is what the conductor may do. This is *when* — the loop that shows a
model the session, reads a typed tool-call list back, runs it, and records the
turn. Nothing here writes a note either: the model chooses, the tools act, and
every pitch still comes out of `compose`.

Five decisions hold this together, and the first is the one that shapes the rest.

**One model call per turn.** Not a loop until the model stops asking for tools,
which is the shape a coding harness has and the obvious thing to reach for. Two
reasons, and the first is a type. `Turn` refuses to carry both calls and an
`llm_error`, because "a turn whose own model call failed ran no tools" — which is
true only while the model call is the turn's *single* point of failure. A second
model call mid-turn can fail after tools have run, and there would be no honest
way to record it. The second is that a turn's length would become the model's
decision: the log's granularity, the user's wait and the turn budget would all
move together with the model's mood. So continuation is explicit instead — the
harness asks again with `trigger="auto"`, which is what that trigger value has
always meant — and the number of automatic continuations is the harness's
decision, in code, like every other budget.

**A turn is (policy, state, what the user just did) — not the transcript.** The
model is sent a system prompt, a digest of the session derived from its own log,
and this turn's input. It is not sent the growing message history. What it needs
to act is the *state* (which spec, which drafts, what was already tried) and the
digest carries that; a transcript would grow without bound, and the tool results
it would replay are summaries of a state the digest describes better. The honest
loss is the exact wording of an earlier turn's tool result — `critique`'s
findings come back as the metrics that miss, not as the sentences that were
written — and that is the part a next decision does not need.

**The system prompt carries policy, not the tool catalogue.** What each tool does
is a `ToolSpec` built from the registry and sent as `tools`; describing them a
second time here would be two copies of one fact, and the copy in a prompt is the
one that drifts. So the prompt says what the conductor *is* — it decides and does
not write music, a refusal is an answer rather than a failure, publish once — and
the catalogue says what the tools do.

**The digest is derived, not appended.** Everything the model is shown about the
session is computed from `Session` at request time, deterministically. That is
what makes replay possible at all: the model is the only non-deterministic part
of a session, so a recorded log plus the state that produced each request is the
whole of it, and nothing has to be remembered separately about what the model was
once told.

**`replay` re-runs the log, and that is how "the log is a faithful record" stops
being a convention.** It dispatches every recorded call in order against a fresh
context. The model is skipped because it is already recorded; what the replay
proves is that the *calls* reproduce the *state*. A replay that includes
`parse_brief` needs a model, or the spec it produced has to be seeded first —
which is exactly why a session stores its own spec rather than only the words the
user typed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Final

from saimc.llm.base import ChatClient, ChatRequest, Message, ToolCall
from saimc.session.models import Session, ToolInvocation, Turn, TurnTrigger
from saimc.session.tools import ToolContext, dispatch, tool_specs

SYSTEM_PROMPT: Final[str] = """\
You are the conductor of a music-writing session. You do not write music: every
note comes from a deterministic engine, and calling the tools you have been given
is the only way to change what the piece sounds like. Your job is to decide what
happens next, and to say why, briefly, in the user's own terms.

- Ask for work, or stop. A reply with no tool call ends the turn. If something is
  worth doing now, call for it; if nothing is, say so in a sentence.
- Never describe music no tool produced. Candidates come back composed and
  measured, so read the measurements rather than imagining what a candidate
  sounds like, and treat a quality finding as a fact you may quote.
- A refusal is an answer, not a failure. It names what to do instead, and it has
  changed nothing — so a different call is a new attempt, and repeating the one
  that was refused gets the same refusal.
- One decision per turn. The user is waiting on it.
- Publishing is final. A session publishes once, and only when the best candidate
  is one worth mastering. The linter is the judge of what is legal and the user
  is the judge of what is good; neither is yours to overrule.

A brief may be specific or open. "90 seconds of calming piano in C at 76 BPM" is
an instruction — draft one candidate and obey it. "Something for a rainy day" is
an invitation — draft several and let the differences between them be the choice
you are offering. Telling the two apart is your first judgement, and it is the
reason you are here rather than a rule table.
"""

MAX_DIGEST_TURNS: Final[int] = 6
"""How many earlier turns the digest shows, most recent last.

Bounded because the digest is sent on every turn: an unbounded one would make
the request grow with the session, and the state — the spec, the drafts, the
verdicts — is what a decision actually turns on. Older turns are summarised by a
count rather than dropped silently, so a model reading the digest knows there is
more it is not being shown.
"""

_SPEC_FIELDS: Final[tuple[str, ...]] = (
    "mood",
    "duration_seconds",
    "tempo_bpm",
    "key",
    "time_signature",
    "instrumentation",
    "seed",
    "humanization",
)
"""The spec's musical fields, in the order the digest lists them.

`request_kind` and `schema_version` are plumbing and are not sent: a model
choosing what to draft next has no decision that turns on either.
"""


def _arguments(call: dict[str, Any]) -> str:
    """A call's arguments, in the order they were sent, as `k=v, k=v`."""
    if not call:
        return ""
    return ", ".join(
        f"{key}={value!r}" if isinstance(value, str) else f"{key}={value}"
        for key, value in call.items()
    )


def _line(invocation: ToolInvocation) -> str:
    """One recorded turn entry, small enough for a digest."""
    rendered = f"{invocation.name}({_arguments(invocation.arguments)})"
    if invocation.ok:
        return rendered
    return f"{rendered} → {invocation.outcome}: {invocation.error_code}"


def _value(name: str, value: Any) -> str:
    """One spec field, rendered the way a reader would say it out loud.

    Two fields are not scalars and are named here rather than coerced.
    `str()` on the instrumentation gives `[InstrumentationEntry(role=
    <VoiceRole.MELODY: 'melody'>, …)]` — a Pydantic repr the model would have
    to parse past to learn that the melody is on the piano, which is the one
    thing the field is for. Everything else is a string, an int or a
    `StrEnum`, where `str()` already reads correctly.
    """
    if name == "duration_seconds":
        return f"{value}s"
    if name == "instrumentation":
        return (
            "[" + ", ".join(f"{entry.role.value}={entry.instrument.value}" for entry in value) + "]"
        )
    return str(value)


def _spec_line(session: Session) -> str:
    if session.spec is None:
        return "spec: not parsed yet — call parse_brief before drafting"
    fields = {
        name: getattr(session.spec, name)
        for name in _SPEC_FIELDS
        if getattr(session.spec, name) is not None
    }
    return "spec: " + ", ".join(f"{name}={_value(name, value)}" for name, value in fields.items())


def _draft_line(draft: Any) -> str:
    """One candidate: how to name it, what it came from, and what it measured.

    The *findings* are reported rather than the raw metrics. A finding is the
    part that says what is wrong with a candidate, which is what a next decision
    is about; the ten measurements behind it are what `critique` is for.

    A revision says which draft it came from, because that is what makes it a
    revision: a model choosing what to do next needs to know that two of the
    drafts on its list are the same piece, or it will treat them as rivals.
    """
    sketch = " sketched" if draft.sketch is not None else ""
    lineage = f" from {draft.parent_id}" if draft.parent_id is not None else ""
    misses = ", ".join(finding.metric for finding in draft.quality.findings())
    return (
        f"  {draft.draft_id} seed={draft.spec.seed}{lineage}{sketch} misses: {misses or 'nothing'}"
    )


def digest(session: Session) -> str:
    """The session, rendered for the model — the state a decision turns on.

    Deterministic and derived: two calls with the same session produce the same
    text, which is what lets a replayed request be the request that was made.
    """
    lines = [f"brief: {session.brief}", _spec_line(session)]
    if session.drafts:
        lines.append(f"drafts ({len(session.drafts)}):")
        lines.extend(_draft_line(draft) for draft in session.drafts)
    else:
        lines.append("drafts (0): none yet")
    if session.verdicts:
        rendered = []
        for verdict in session.verdicts:
            parts = [verdict.draft_id]
            if verdict.value is not None:
                parts.append(verdict.value)
            if verdict.feedback:
                parts.append(f"{verdict.feedback!r}")
            rendered.append(" ".join(parts))
        lines.append("verdicts: " + "; ".join(rendered))
    lines.append(
        "published: not yet"
        if session.finalized_job_id is None
        else f"published: job {session.finalized_job_id}"
    )

    shown = session.turns[-MAX_DIGEST_TURNS:]
    if shown:
        earlier = len(session.turns) - len(shown)
        lines.append(f"turns ({len(session.turns)} in all):")
        if earlier:
            lines.append(f"  … {earlier} earlier turn(s) not shown")
        for turn in shown:
            if turn.llm_error is not None:
                body = f"the model call failed: {turn.llm_error.error_code}"
            else:
                calls = ", ".join(_line(call) for call in turn.calls)
                body = calls or "said nothing and called nothing"
            lines.append(f"  {turn.trigger} → {body}")
    return "\n".join(lines)


def _user_message(session: Session, *, trigger: TurnTrigger, message: str) -> str:
    """This turn's input: the state, then what asked for the turn.

    Three triggers, three sentences, and each says something the state above
    does not. The brief is already the digest's first line, so this turn's line
    is an *instruction* about it rather than a second copy of it — the same
    reason the user's own words are quoted back here instead of being left in
    the log as a past turn.

    The `auto` branch is the only one that mentions publishing, because an
    automatic turn is only ever asked for by a caller that asked for
    auto-finalize: it is the harness's continuation after a pass that published
    nothing, and the one thing worth saying about it is that the piece may be
    finished. Every other turn is the user's, and a user's turn is not the
    place to be told to hurry.
    """
    if trigger == "brief":
        asked = "This is the session's first turn: work the brief above into candidates."
    elif trigger == "message":
        asked = f"The user says: {message}"
    else:
        asked = (
            "No one has spoken. Review where this session stands and take it "
            "forward: say what you did, and call for whatever is worth doing next. "
            "If a candidate is ready to be mastered, publish it."
        )
    return f"{digest(session)}\n\n{asked}"


async def take_turn(
    ctx: ToolContext,
    client: ChatClient,
    *,
    trigger: TurnTrigger,
    message: str = "",
) -> Turn:
    """Run one turn: one model call, and the tools it asks for, in order.

    Returns the recorded `Turn` and appends it to the session. The session is
    saved once, here, rather than by the caller: a turn is a decision the user
    has already seen the result of, so the record of it has to be durable before
    the caller renders anything.

    A model call that fails is recorded as a turn with no calls at all — nothing
    ran, and the type refuses to claim otherwise. `client.chat` raising is *not*
    caught: the adapter's contract is to answer with a named `LLMError`, so a
    raise is a defect, and folding it into a recorded `llm_error` would make a
    broken adapter indistinguishable from an unreachable host.
    """
    ctx.begin_turn()
    started = time.monotonic()
    request = ChatRequest(
        messages=(
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user", content=_user_message(ctx.session, trigger=trigger, message=message)
            ),
        ),
        tools=tool_specs(ctx.budget),
        request_id=f"{ctx.session.session_id}:{len(ctx.session.turns)}",
    )
    result = await client.chat(request)
    latency_ms = int((time.monotonic() - started) * 1000)

    if result.error is not None:
        turn = Turn(
            created_at=datetime.now(UTC),
            trigger=trigger,
            latency_ms=latency_ms,
            llm_error=result.error,
        )
    else:
        # In the order the model asked for them, which is the order it will read
        # them back in: a turn whose calls ran out of order would replay wrong.
        calls = tuple([await dispatch(call, ctx) for call in result.tool_calls])
        turn = Turn(
            created_at=datetime.now(UTC),
            trigger=trigger,
            narration=result.content.strip(),
            calls=calls,
            model=result.extra.get("model_identifier"),
            latency_ms=latency_ms,
        )

    ctx.session.turns.append(turn)
    ctx.sessions.save(ctx.session)
    return turn


async def replay(turns: Iterable[Turn], ctx: ToolContext) -> list[ToolInvocation]:
    """Re-run recorded tool calls, in order, and return what they answer now.

    This is the check on the log rather than on the model: the model is already
    recorded, and what a replay proves is that the calls it made reproduce the
    state they produced. Against a fresh session seeded with the same spec, and
    with the renderer and the broker stubbed, the drafts come back with the same
    plans and the same performance hashes — which is the whole claim that a
    session is replayable.

    A turn that failed its own model call asked for nothing and contributes
    nothing. A `parse_brief` call is re-run against `ctx.llm`, so a replay that
    covers the first turn needs a model or a seeded spec; with neither it
    refuses, visibly, in the returned invocation rather than silently.

    The `ToolCall` is rebuilt from the record rather than passed as it stands,
    and the arguments are copied. `Turn.calls` holds `ToolInvocation`s — the
    outcomes, which is what a log is — and `dispatch` takes the *ask*; the two
    share `name` and `arguments`, so the tempting version type-checks in a
    duck-typed language and is wrong twice: it hands a handler the record it is
    about to overwrite, and once the two types diverge it hands it the wrong
    one.
    """
    return [
        await dispatch(ToolCall(name=call.name, arguments=dict(call.arguments)), ctx)
        for turn in turns
        for call in turn.calls
    ]


__all__ = ["MAX_DIGEST_TURNS", "SYSTEM_PROMPT", "digest", "replay", "take_turn"]
