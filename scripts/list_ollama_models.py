"""List the text models available on the configured Ollama host.

Roadmap §10 #3: before the Phase 1 model is pinned, query the
configured Ollama Cloud host for currently available text models,
discard any that fail the license/acceptable-use gate, and benchmark
the rest. This helper performs that query and prints a paste-ready
candidate block for `MODELS.md`.

Usage:
    OLLAMA_BASE_URL=... OLLAMA_API_KEY=... python scripts/list_ollama_models.py
"""

from __future__ import annotations

import os
import sys

import ollama


def main() -> int:
    base_url = (os.environ.get("OLLAMA_BASE_URL") or "").rstrip("/")
    if not base_url:
        print("OLLAMA_BASE_URL is not set; nothing to query.", file=sys.stderr)
        return 1
    api_key = os.environ.get("OLLAMA_API_KEY") or None

    client = ollama.Client(
        host=base_url, headers={"Authorization": f"Bearer {api_key}"} if api_key else {}
    )
    response = client.list()
    models = [m.model for m in response.models if m.model is not None]
    if not models:
        print(f"No models reported by {base_url}.", file=sys.stderr)
        return 1

    print(f"# Text models reported by {base_url} (paste into MODELS.md, then apply")
    print("# the license/acceptable-use gate and the 100-prompt benchmark per §10 #3)")
    for name in sorted(models):
        print(f"- Model identifier: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
