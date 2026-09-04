"""Engine output sidecar persistence.

When `compose_stage` finishes, it serializes the `EngineOutput` to a
sidecar JSON file under the job directory. Later stages (audio,
sheet, animation) load it back instead of re-running the composer.

The sidecar is canonical JSON (sorted keys, no insignificant
whitespace) so its sha256 is deterministic for the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saimc.canonical import canonical_dumps
from saimc.compose.engine import EngineOutput


def write_engine_output(path: Path, output: EngineOutput) -> None:
    """Atomically write `output` to `path` as canonical JSON."""
    payload: dict[str, Any] = output.to_sidecar()
    text = canonical_dumps(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_engine_output(path: Path) -> EngineOutput:
    """Read `output` back from the sidecar. Raises `FileNotFoundError` if missing."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return EngineOutput.from_sidecar(payload)


__all__ = ["read_engine_output", "write_engine_output"]
