"""LLMClient — the provider-neutral interface every prompt-parsing adapter
implements. See `docs/roadmap.md` §3 (provider-neutral design).

Phase 1 supplies one concrete adapter: `OllamaAdapter` (saimc/llm/ollama.py).
A future commercial provider would require its own adapter behind this same
interface; provider-specific auth and request mapping stay inside each adapter.

The interface is intentionally narrow: one method, `parse`, which takes a
user prompt and returns either a validated CompositionSpec or a structured
error. The transport (HTTP, SDK, local file), the auth (Bearer, none), and
the JSON-enforcement strategy (schema-mode, prompted-JSON, retries) all live
inside the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


__all__ = ["LLMClient", "ParseRequest", "ParseResult"]
