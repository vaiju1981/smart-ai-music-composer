"""The interactive session: a brief, a conductor, drafts, and the user's verdict.

This package is the harness laid over the one-shot pipeline. It is the only
place a model drives a loop, and it is deliberately the top of the import
graph: `compose/`, `render/` and `release/` may not import it
(`tests/unit/test_import_boundary.py`), because the engine must stay the
only writer of notes and the only thing a determinism claim is made about.

The layering below it is one-directional:

- `saimc.llm` supplies transport (`ChatClient`) and prompt-to-spec parsing
  (`LLMClient.parse`). Neither knows a session exists.
- `saimc.compose` and `saimc.quality` are deterministic in what they are
  given, which is what makes a draft a record rather than a copy.
- `saimc.jobs` is the render pipeline and its 9-state machine, untouched by
  any of this: a sketch is not a job, and a finalized piece becomes one.

What lives here: `models` (the records), `store` (their persistence),
`arbiter` (one order over drafts, and the revision ratchet), `deltas` (the
typed requests a user or an agent may make, and the applier), `conductor`
(the loop), `tools` (what the conductor may call), `api` (the HTTP surface).
"""

from __future__ import annotations
