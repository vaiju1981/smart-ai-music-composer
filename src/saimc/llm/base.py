"""The provider-neutral LLM interfaces. See `docs/roadmap.md` §3.

Two protocols live here, and they are *layered* rather than peers:

- `LLMClient.parse` takes a user prompt and returns either a validated
  CompositionSpec or a structured error. It is not a transport primitive —
  it is a chat call plus the reading of the reply as a music spec, and the
  repair loop and fallback around it live in `saimc/parser.py`.
- `ChatClient.chat` is the layer underneath: messages and tool schemas in,
  tool calls out. It carries no music type at all, because the session
  conductor's vocabulary belongs to `saimc/session/` and not to the
  transport package.

Keeping them separate is what lets `parse` later be re-expressed in terms of
`chat` without a caller noticing, and what keeps the music layers outside
this package (§3's provider-neutrality rule, enforced by the AST test in
`tests/unit/test_import_boundary.py`).

Phase 1 supplies one concrete adapter for both: `OllamaAdapter`
(saimc/llm/ollama.py). A future commercial provider would require its own
adapter behind these same interfaces; provider-specific auth and request
mapping stay inside each adapter. The transport (HTTP, SDK, local file), the
auth (Bearer, none), and the JSON-enforcement strategy (schema-mode,
prompted-JSON, retries) all live inside the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from saimc.spec import CompositionSpec, SpecError


@dataclass(frozen=True)
class ParseRequest:
    """The input to LLMClient.parse.

    `system_prompt` is provided by the adapter (or its config); clients do
    not get to inject it. `prompt` is the user-typed input.
    """

    prompt: str
    request_id: str
    """Opaque correlation token for benchmark/run logging; not used by the adapter logic."""

    previous_error: str | None = None
    """Why the previous attempt failed (error code + message), for the §6
    repair loop. None on the first attempt; adapters fold this into the
    system prompt so the model can fix what went wrong instead of
    re-rolling a blind retry."""


@dataclass(frozen=True)
class ParseResult:
    """The output of LLMClient.parse. Exactly one of `spec` or `error` is set."""

    parser_source: str
    """`llm` | `fallback` | `hybrid` — recorded in the job metadata and benchmark output."""

    attempts: int = 0
    """Number of LLM round-trips used to reach this result (0 for fallback-only)."""

    structured_output: bool = False
    """True if the adapter used the host's native schema-enforced structured-output
    capability; False if it used prompted JSON. Recorded in metadata per §3."""

    capability_probe: str | None = None
    """Free-form capability-probe result line (`schema_native | prompted_json | error:...`)
    for logging and benchmark diagnostics. None if no probe was run."""

    first_attempt: Literal["valid", "invalid"] | None = None
    """What the *first* attempt read — set by `saimc.parser.parse_prompt`, not by an adapter.

    **Three states, not two.** `"valid"` and `"invalid"` are the two the field's
    name implies, and `None` is the third: no model output was read at all. It is
    the value a run gets when every attempt failed to reach the host, and it is
    not the same thing as `"invalid"` — §8's first bar is about a model's first
    *response*, and a run in which no model ever answered cannot be scored
    against it. Recording `"invalid"` there would charge the model for the
    network's failure, which is the one thing that makes the bar meaningless on a
    flaky host.

    A typed field rather than an `extra` key, because `extra` is documented above
    as the free-form *diagnostic* channel and this value drives a scored release
    bar; and rather than an inference from `attempts` plus the error code,
    because that rule would live outside the loop that witnessed the attempt. The
    loop is the witness; this is where its testimony goes.

    Adapters leave it alone. An adapter does not know whether its call was the
    first one, and a `ParseResult` built by one is an attempt rather than a run.
    """

    spec: CompositionSpec | None = None
    error: SpecError | None = None
    extra: dict[str, str] = field(default_factory=dict)
    """Free-form metadata (latency_ms, prompt_chars, model_identifier, …). Strings only."""

    def __post_init__(self) -> None:
        if (self.spec is None) == (self.error is None):
            raise ValueError("ParseResult must set exactly one of `spec` or `error`")


@runtime_checkable
class LLMClient(Protocol):
    """Provider-neutral LLM access interface.

    Implementations are constructed once per process (so the startup
    capability probe runs at most once) and may be reused concurrently
    only if they document themselves as thread-safe.
    """

    async def parse(self, request: ParseRequest) -> ParseResult: ...

    async def aclose(self) -> None:
        """Release any held resources (HTTP client, SDK handles)."""
        ...


@dataclass(frozen=True)
class LLMError:
    """A transport-level failure, with no music semantics.

    `SpecError` cannot serve here: it lives in `saimc/spec.py` and its
    `stage` is the parse pipeline's vocabulary (`parsing`/`validating`/
    `fallback`), whereas this describes a *turn*. The codes that mean the
    same thing in both vocabularies are spelled the same way
    (`llm_unreachable`, `llm_invalid_response`, `llm_not_configured`), so a
    reader moving between the two sees one vocabulary rather than two.
    """

    error_code: str
    message: str


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for. A name and its arguments, nothing more.

    There is deliberately no call id: Ollama does not issue one, and a tool
    result is correlated by `Message.tool_name` and by order. Inventing an
    id here would imply a correlation the wire does not carry.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """One tool offered to the model, in provider-neutral form.

    `parameters` is a JSON Schema object describing the arguments. The
    *content* names musical tools; the type does not, which is what keeps
    this module free of music types.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One turn of a conversation, in the shape the wire uses.

    A `tool` message answers a call and names the tool it answers. An
    `assistant` message may carry `tool_calls` *and* empty content — that is
    the normal shape of a tool-calling turn, and it is re-sent verbatim in
    the history, so it has to be representable.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_name:
            raise ValueError("a tool message must name the tool it answers")
        if self.role != "tool" and self.tool_name is not None:
            raise ValueError(f"only a tool message may carry tool_name (role={self.role!r})")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError(f"only an assistant message may carry tool_calls (role={self.role!r})")


@dataclass(frozen=True)
class ChatRequest:
    """The input to ChatClient.chat.

    `messages` carries the system prompt as its first entry, which is the
    opposite of `ParseRequest`'s convention and is deliberate. The parse
    adapter's system prompt is a fixed contract whose wording the parse
    benchmarks (§10) are scored against, so callers may not inject it. The
    conductor's system prompt is the *harness's* policy — which tools exist
    and what the rules are — and it changes whenever the tool catalogue
    does, so it cannot be baked into the adapter.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    request_id: str = ""
    """Opaque correlation token for benchmark/run logging; not used by the adapter logic."""

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a chat request needs at least one message")


@dataclass(frozen=True)
class ChatResult:
    """The output of ChatClient.chat. Either a reply or a named error.

    A successful reply may carry content, tool calls, or both — a model that
    answers in prose when tools were offered is a normal outcome, not an
    error, and the conductor is what decides whether prose is acceptable.
    What cannot happen is both a reply and an error.
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    error: LLMError | None = None
    extra: dict[str, str] = field(default_factory=dict)
    """Free-form metadata (latency_ms, model_identifier, …). Strings only."""

    def __post_init__(self) -> None:
        if self.error is not None and (self.content or self.tool_calls):
            raise ValueError("a failed chat result cannot also carry a reply")

    @property
    def ok(self) -> bool:
        return self.error is None


@runtime_checkable
class ChatClient(Protocol):
    """Provider-neutral tool-calling chat access.

    The counterpart to `LLMClient`, one layer down: `LLMClient.parse` is a
    chat call plus the reading of the reply as a CompositionSpec, so the two
    are layered rather than peers. Implementations are constructed once per
    process and must be reusable concurrently only if they document
    themselves as thread-safe.
    """

    async def chat(self, request: ChatRequest) -> ChatResult: ...

    async def aclose(self) -> None:
        """Release any held resources (HTTP client, SDK handles)."""
        ...


__all__ = [
    "ChatClient",
    "ChatRequest",
    "ChatResult",
    "LLMClient",
    "LLMError",
    "Message",
    "ParseRequest",
    "ParseResult",
    "ToolCall",
    "ToolSpec",
]
