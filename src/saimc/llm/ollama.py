"""OllamaAdapter — Phase 1 LLM access.

Talks to an Ollama host (Cloud or self-hosted) via the `ollama` Python SDK.
Per `docs/roadmap.md` §3:

- Auth is env-var driven: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`,
  `OLLAMA_API_KEY`. The API key is optional and only attached when present.
- Native schema-enforced structured output is the preferred mode when the
  configured host supports it; otherwise we fall back to prompted JSON with
  the same local validation + 2-repair policy.
- The mode and probe result are recorded in `ParseResult` for the job and
  benchmark metadata.

The adapter does NOT depend on a real Ollama host to import or unit-test —
the SDK is imported lazily so tests can run with no network access and no
OLLAMA_* environment. A live Ollama Cloud test is a release-gate task.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from saimc.llm.base import (
    ChatRequest,
    ChatResult,
    LLMError,
    Message,
    ParseRequest,
    ParseResult,
    ToolCall,
    ToolSpec,
)
from saimc.spec import CompositionSpec, SpecError

SCHEMA_ENFORCED_MARKER = "schema_native"
PROMPTED_JSON_MARKER = "prompted_json"
PROBE_FAILURE_MARKER = "error:probe_failed"

_DEFAULT_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 10.0
_PROBE_PROMPT = "Respond with the single word: pong"


class OllamaAdapter:
    """Async LLMClient and ChatClient backed by an Ollama HTTP host.

    Constructed once per process; the startup capability probe runs once in
    `__init__` (it does NOT make a real chat call — it issues a tiny throwaway
    request and checks the host's response shape to decide whether to use
    `format=` schema enforcement or prompted JSON).

    Construction is cheap; `aclose()` releases the underlying httpx client.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or "").rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL") or ""
        self._api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY") or None
        self._timeout_s = timeout_s
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=timeout_s,
            headers=self._auth_headers(),
        )
        self._structured_output: bool | None = None
        self._probe_result: str | None = None

        if not self._base_url:
            raise ValueError("OllamaAdapter requires OLLAMA_BASE_URL (or pass base_url=).")
        if not self._model:
            raise ValueError("OllamaAdapter requires OLLAMA_MODEL (or pass model=).")

    @property
    def model_identifier(self) -> str:
        return self._model

    @property
    def structured_output(self) -> bool | None:
        """None until `probe_capability()` has been called."""
        return self._structured_output

    @property
    def probe_result(self) -> str | None:
        return self._probe_result

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def probe_capability(self) -> bool:
        """Issue a tiny throwaway request to detect schema-enforced output support.

        The probe sends the `format` parameter set to a minimal JSON schema and
        checks that the response is valid JSON matching the schema's shape.
        On failure, the adapter falls back to prompted JSON.

        Sets `structured_output` and `probe_result`; returns True iff native
        schema enforcement is supported.
        """
        probe_schema = {
            "type": "object",
            "properties": {"pong": {"type": "string"}},
            "required": ["pong"],
        }
        try:
            resp = await self._http.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": _PROBE_PROMPT}],
                    "format": probe_schema,
                    "stream": False,
                },
                timeout=_PROBE_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("message") or {}).get("content") or ""
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("pong"):
                self._structured_output = True
                self._probe_result = SCHEMA_ENFORCED_MARKER
                return True
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            pass
        self._structured_output = False
        self._probe_result = PROMPTED_JSON_MARKER
        return False

    async def parse(self, request: ParseRequest) -> ParseResult:
        if self._structured_output is None:
            await self.probe_capability()

        schema = CompositionSpec.model_json_schema()
        messages = self._build_messages(
            request.prompt,
            schema,
            prompted=not self._structured_output,
            previous_error=request.previous_error,
        )

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if self._structured_output:
            body["format"] = schema

        t0 = time.perf_counter()
        try:
            resp = await self._http.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # A failed request still took time, and it is the same reading the
            # other four branches carry: the elapsed time of the attempt.
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return ParseResult(
                parser_source="llm",
                attempts=1,
                structured_output=bool(self._structured_output),
                capability_probe=self._probe_result,
                error=SpecError(
                    error_code="llm_unreachable",
                    message=f"Ollama HTTP error: {exc}",
                    stage="parsing",
                    attempts=1,
                ),
                extra={"latency_ms": str(latency_ms)},
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        try:
            payload = resp.json()
            content = (payload.get("message") or {}).get("content") or ""
        except (json.JSONDecodeError, ValueError) as exc:
            return ParseResult(
                parser_source="llm",
                attempts=1,
                structured_output=bool(self._structured_output),
                capability_probe=self._probe_result,
                error=SpecError(
                    error_code="llm_invalid_response",
                    message=f"Could not parse Ollama response: {exc}",
                    stage="parsing",
                    attempts=1,
                ),
                extra={"latency_ms": str(latency_ms)},
            )

        try:
            parsed_obj = json.loads(content)
        except json.JSONDecodeError as exc:
            return ParseResult(
                parser_source="llm",
                attempts=1,
                structured_output=bool(self._structured_output),
                capability_probe=self._probe_result,
                error=SpecError(
                    error_code="llm_invalid_json",
                    message=f"Model returned non-JSON content: {exc}",
                    stage="parsing",
                    attempts=1,
                ),
                extra={"latency_ms": str(latency_ms)},
            )

        try:
            spec = CompositionSpec.model_validate(parsed_obj)
        except Exception as exc:
            return ParseResult(
                parser_source="llm",
                attempts=1,
                structured_output=bool(self._structured_output),
                capability_probe=self._probe_result,
                error=SpecError(
                    error_code="schema_invalid",
                    message=f"Model output failed schema validation: {exc}",
                    stage="validating",
                    attempts=1,
                ),
                extra={"latency_ms": str(latency_ms)},
            )

        return ParseResult(
            parser_source="llm",
            attempts=1,
            structured_output=bool(self._structured_output),
            capability_probe=self._probe_result,
            spec=spec,
            extra={
                "latency_ms": str(latency_ms),
                "model_identifier": self._model,
                "request_id": request.request_id,
            },
        )

    async def chat(self, request: ChatRequest) -> ChatResult:
        """One tool-calling turn against the Ollama host.

        No capability probe gates this. Schema enforcement is a *mode* the
        host either has or lacks, so `parse` has to detect it before it can
        choose a body; tool calling has no alternative mode to fall back to,
        and whether a given model honours `tools` is answered by the reply
        itself — prose where calls were offered is a normal outcome the
        conductor decides what to do with, not a transport error. A host
        that *rejects* the parameter answers with an HTTP error, which is
        reported as such rather than guessed at from a body shape.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [self._wire_message(message) for message in request.messages],
            "stream": False,
        }
        if request.tools:
            body["tools"] = [self._wire_tool(tool) for tool in request.tools]

        t0 = time.perf_counter()
        try:
            resp = await self._http.post(f"{self._base_url}/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return ChatResult(error=LLMError("llm_unreachable", f"Ollama HTTP error: {exc}"))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        extra = {"latency_ms": str(latency_ms), "model_identifier": self._model}

        try:
            payload = resp.json()
            message = payload.get("message") or {}
            content = message.get("content") or ""
            raw_calls = message.get("tool_calls") or []
        except (json.JSONDecodeError, ValueError, AttributeError) as exc:
            return ChatResult(
                error=LLMError("llm_invalid_response", f"Could not parse Ollama response: {exc}"),
                extra=extra,
            )

        try:
            calls = tuple(self._read_tool_call(raw) for raw in raw_calls)
        except ValueError as exc:
            return ChatResult(
                error=LLMError("tool_call_malformed", f"Unreadable tool call: {exc}"),
                extra=extra,
            )

        return ChatResult(content=content, tool_calls=calls, extra=extra)

    @staticmethod
    def _wire_message(message: Message) -> dict[str, Any]:
        wire: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_name is not None:
            wire["tool_name"] = message.tool_name
        if message.tool_calls:
            wire["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in message.tool_calls
            ]
        return wire

    @staticmethod
    def _wire_tool(tool: ToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    @staticmethod
    def _read_tool_call(raw: Any) -> ToolCall:
        """Read one call from the wire, refusing rather than defaulting.

        The documented shape puts a JSON object in `arguments`, but a model
        that emits the arguments as a JSON *string* is common enough that
        reading only the object form would drop its calls on the floor.
        Unreadable arguments raise instead of becoming `{}`: a tool run with
        silently empty arguments would be a wrong action, not a missing one.
        """
        function = (raw or {}).get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"call has no tool name: {raw!r}")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name}: arguments are not JSON ({exc})") from exc
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"{name}: arguments are not an object: {arguments!r}")
        return ToolCall(name=name, arguments=arguments)

    @staticmethod
    def _build_messages(
        prompt: str,
        schema: dict[str, Any],
        *,
        prompted: bool,
        previous_error: str | None = None,
    ) -> list[dict[str, str]]:
        sys = (
            "You convert natural-language music requests into a strict JSON "
            "CompositionSpec. Output ONLY the JSON object. Do not wrap it in "
            "markdown fences. Do not include commentary. The schema is enforced. "
            "Preserve every explicitly requested instrument, assigning exactly one "
            "melody, at most one bass, at most one drum_set percussion part, and up "
            "to two complementary harmony parts. Requests for a professional, "
            "cinematic, natural, polished, or human performance should use "
            "humanization='expressive'."
        )
        if prompted:
            sys += " The JSON object MUST match this schema exactly: " + json.dumps(
                schema, separators=(",", ":")
            )
        if previous_error:
            sys += (
                f" Your previous attempt was rejected: {previous_error}. "
                "Return a corrected JSON object that fixes exactly that problem."
            )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ]


__all__ = ["PROMPTED_JSON_MARKER", "SCHEMA_ENFORCED_MARKER", "OllamaAdapter"]
