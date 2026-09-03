"""Top-level prompt-to-spec orchestration.

Per `docs/roadmap.md` §6 ("Repair/retry policy"):

1. The LLM adapter is invoked once.
2. If its output fails schema validation, up to 2 repair attempts are made,
   each re-prompting with the schema-in-context error feedback.
3. If still invalid, the deterministic fallback parser is invoked for
   in-vocabulary prompts; out-of-vocabulary prompts fail with a structured
   error and the prompt is returned to the user for editing.

This module also exposes `ParserSource` for benchmark and job metadata.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from saimc.llm.fallback import parse_fallback
from saimc.spec import CompositionSpec, SpecError

if TYPE_CHECKING:
    from saimc.llm.base import LLMClient, ParseResult

MAX_LLM_ATTEMPTS: int = 3
"""Initial attempt + 2 repairs, per the §6 retry policy."""


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
    """
    from saimc.llm.base import ParseRequest as _PR
    from saimc.llm.base import ParseResult as _PRs

    request = _PR(prompt=prompt, request_id=request_id)

    last_error: SpecError | None = None
    last_result: ParseResult | None = None

    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        result = await client.parse(request)
        result = dataclasses.replace(result, attempts=attempt)
        last_result = result
        if result.spec is not None:
            return result
        last_error = result.error
        if result.error is None:
            return dataclasses.replace(
                result,
                error=SpecError(
                    error_code="internal_no_error_no_spec",
                    message="Adapter returned neither spec nor error.",
                    stage="parsing",
                    attempts=attempt,
                ),
            )

    fallback_outcome = parse_fallback(prompt)
    if isinstance(fallback_outcome, CompositionSpec):
        return _PRs(
            parser_source="fallback" if last_result is None else "hybrid",
            attempts=MAX_LLM_ATTEMPTS,
            structured_output=bool(last_result.structured_output) if last_result else False,
            capability_probe=last_result.capability_probe if last_result else None,
            spec=fallback_outcome,
            extra={
                "fallback_used": "true",
                "llm_error_code": last_error.error_code if last_error else "",
            },
        )

    assert last_error is not None  # guaranteed by the loop above
    return _PRs(
        parser_source="llm",
        attempts=MAX_LLM_ATTEMPTS,
        structured_output=bool(last_result.structured_output) if last_result else False,
        capability_probe=last_result.capability_probe if last_result else None,
        error=last_error,
        extra={
            "fallback_error_code": fallback_outcome.error_code,
            "fallback_error_stage": fallback_outcome.stage,
        },
    )


__all__ = ["MAX_LLM_ATTEMPTS", "parse_prompt"]
