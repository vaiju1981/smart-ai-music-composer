"""Top-level prompt-to-spec orchestration.

Per `docs/roadmap.md` §6 ("Repair/retry policy"):

1. The LLM adapter is invoked once.
2. If its output fails schema validation, up to 2 repair attempts are made,
   each re-prompting with the schema-in-context error feedback.
3. If still invalid, the deterministic fallback parser is invoked for
   in-vocabulary prompts; out-of-vocabulary prompts fail with a structured
   error and the prompt is returned to the user for editing.

This module is also the only place that can answer §8's first parse bar — "≥ 98%
of first responses are valid JSON matching `CompositionSpec`" — because it is
the only place that sees attempt 1. The loop returns at the first valid attempt
and rebinds `last_result` on every iteration, so after it exits the first
attempt's own result is gone; the loop is the witness, and
`ParseResult.first_attempt` is where its testimony is written down. See
`NO_COMPLETION_ERROR_CODES` for the third state that field needs and why.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

from saimc.llm.fallback import parse_fallback
from saimc.spec import CompositionSpec, SpecError

if TYPE_CHECKING:
    from saimc.llm.base import LLMClient, ParseResult

MAX_LLM_ATTEMPTS: int = 3
"""Initial attempt + 2 repairs, per the §6 retry policy."""

NO_COMPLETION_ERROR_CODES: frozenset[str] = frozenset(
    {
        "llm_unreachable",
        "llm_not_configured",
        "llm_invalid_response",
    }
)
"""The failures under which **no model output was read at all**.

Each of the three is about the call rather than about the reply: the host could
not be reached, no host was configured, or the response body was not a model's
response. None of them is a model answering badly, so none of them can be
scored against §8's first-response bar — and that distinction is the whole
reason `ParseResult.first_attempt` has a third state. A run against an
unreachable host reports *not measured* rather than 0%, because charging the
model for the network's failure is what would make the bar meaningless on a
flaky host.

The other two codes the adapter can emit are deliberately absent, and both mean
the model *did* answer: `llm_invalid_json` (something was said, and it was not
JSON) and `schema_invalid` (JSON was said, and it did not describe a piece).
Those are `"invalid"`, which is a judgement about a response that exists.

The set is public because it is the rule a scored release bar rests on, and
`tests/unit/test_parser.py` ratchets it against the codes the adapters in this
repo can actually emit — so a fifth adapter branch fails a named test until it
is classified here rather than silently reporting 0%.
"""


async def parse_prompt(client: LLMClient, prompt: str, request_id: str) -> ParseResult:
    """Parse a user prompt into a CompositionSpec.

    Strategy:
      1. Call the LLM adapter once with the prompt.
      2. If the result has a schema-invalid error, retry up to 2 times
         (total 3 attempts) with the schema + last error appended to the
         system prompt.
      3. If all attempts fail with a schema-invalid error, try the
         deterministic fallback parser for in-vocabulary requests.
      4. Otherwise return the final error from the LLM attempts.

    The returned `ParseResult.parser_source` is one of:
      - `"llm"` if any LLM attempt produced a schema-valid spec.
      - `"fallback"` if only the fallback produced a valid spec.
      - `"hybrid"` if LLM produced an invalid spec but fallback succeeded.
      - `"llm"` if even the fallback failed (the error is then returned
         with `parser_source="llm"` and the original LLM error preserved).

    `ParseResult.first_attempt` reports what the *first* attempt read, whatever
    the run ended as: `"invalid"` when the model answered and the answer did not
    parse into a spec, `None` when no model output was read at all
    (`NO_COMPLETION_ERROR_CODES`). It is read from attempt 1's own result inside
    the loop, because the loop returns early on success and nothing after it can
    see that result again.
    """
    from saimc.llm.base import ParseRequest as _PR
    from saimc.llm.base import ParseResult as _PRs

    request = _PR(prompt=prompt, request_id=request_id)

    first_attempt: Literal["valid", "invalid"] | None = None
    last_error: SpecError | None = None
    last_result: ParseResult | None = None

    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        # Repairs are not blind retries: attempt N>1 carries what the
        # previous attempt got wrong so the model can fix it.
        if attempt > 1 and last_error is not None:
            request = dataclasses.replace(
                request,
                previous_error=(f"{last_error.error_code}: {last_error.message}"),
            )
        result = await client.parse(request)
        result = dataclasses.replace(result, attempts=attempt)
        last_result = result
        # `attempt == 1` rather than a None sentinel: None is now a value this
        # field takes, so "not yet set" must not be spelled the same way.
        if attempt == 1:
            first_attempt = _first_attempt(result)
        if result.spec is not None:
            return dataclasses.replace(result, first_attempt=first_attempt)
        last_error = result.error

    fallback_outcome = parse_fallback(prompt)
    if isinstance(fallback_outcome, CompositionSpec):
        return _PRs(
            parser_source="fallback" if last_result is None else "hybrid",
            attempts=MAX_LLM_ATTEMPTS,
            structured_output=bool(last_result.structured_output) if last_result else False,
            capability_probe=last_result.capability_probe if last_result else None,
            first_attempt=first_attempt,
            spec=fallback_outcome,
            extra={
                "fallback_used": "true",
                "llm_error_code": last_error.error_code if last_error else "",
            },
        )

    assert last_error is not None  # guaranteed by the loop above
    # When the LLM kept producing schema-invalid output and the
    # fallback rejected the prompt as out-of-vocabulary, the actionable
    # answer for the user is the vocabulary constraint — not the LLM's
    # plumbing error. Surface the fallback rejection (which names the
    # Phase 1 mood keywords) with the LLM error preserved in extra.
    if last_error.error_code == "schema_invalid" and fallback_outcome.error_code in (
        "out_of_vocabulary",
        "empty_prompt",
    ):
        return _PRs(
            parser_source="llm",
            attempts=MAX_LLM_ATTEMPTS,
            structured_output=bool(last_result.structured_output) if last_result else False,
            capability_probe=last_result.capability_probe if last_result else None,
            first_attempt=first_attempt,
            error=fallback_outcome,
            extra={
                "llm_error_code": last_error.error_code,
                "llm_error_message": last_error.message,
            },
        )
    return _PRs(
        parser_source="llm",
        attempts=MAX_LLM_ATTEMPTS,
        structured_output=bool(last_result.structured_output) if last_result else False,
        capability_probe=last_result.capability_probe if last_result else None,
        first_attempt=first_attempt,
        error=last_error,
        extra={
            "fallback_error_code": fallback_outcome.error_code,
            "fallback_error_stage": fallback_outcome.stage,
        },
    )


def _first_attempt(result: ParseResult) -> Literal["valid", "invalid"] | None:
    """Read one attempt's result as the three states `first_attempt` takes.

    A spec is `"valid"` whatever route produced it. An error is `"invalid"`
    unless its code is one of `NO_COMPLETION_ERROR_CODES`, in which case no
    model output was read and the honest answer is `None` — not `"invalid"`,
    which would score the network.
    """
    if result.spec is not None:
        return "valid"
    assert result.error is not None  # ParseResult's own invariant
    return None if result.error.error_code in NO_COMPLETION_ERROR_CODES else "invalid"


__all__ = ["MAX_LLM_ATTEMPTS", "NO_COMPLETION_ERROR_CODES", "parse_prompt"]
