# Smart AI Music Composer — Roadmap & Architecture

Status: **Phase 1 architecture approved.** The implementation and release gates in §10 must pass before an MVP release is published.

## 1. Goal

Prompt-driven music composition app. A user types something like:

- "build me a 5 min electrifying piano piece"
- "5 min of calming / sleep music"
- "Canon in D, but in C"
- "Canon in C using Indian classical instruments"

...and gets back a matched set of outputs: audio, engraved sheet music, and a synced animation (piano-roll style), all traceably derived from one underlying symbolic representation. The outputs share the same musical content; renderer-specific timing and formatting differences are measured by the acceptance criteria in §8.

## 2. Pipeline (source of truth = symbolic representation)

```
Prompt
  │
  ▼
1. Prompt Parser
   - Primary:   Ollama chat with schema-enforced structured output → CompositionSpec
                (also locally validated against the same versioned schema)
   - Fallback:  Deterministic regex/keyword parser for the small Phase 1
                mood vocabulary; used if Ollama is unavailable, rate-limited,
                or fails schema validation after 2 repairs.
  │
  ▼
2. Strategy Selector
   - If spec.request_kind == "famous_piece"   → Known-Piece Matcher (Phase 2)
   - If spec.request_kind == "mood_generation" → Composition Engine
   Famous-piece handling is Phase 2 — v1 accepts mood-based prompts only.
  │
  ▼
3. Composition Engine
   - rule-based generation (music21), optional cloud-LLM creative assist
     for melodic/harmonic seeds (Phase 2)
   - produces TWO artifacts, explicitly:
       NotationScore:   pitches, durations, measures, dynamics (the engraved surface)
       PerformancePlan: realized timestamps, velocities, articulations,
                        humanization (the playback surface)
   - theory-linter pass: range, voice-leading, rhythm sanity, complete measures
  │
  ▼
4. Renderers, each consuming its appropriate artifact:
   4a. Audio     → PerformancePlan → MIDI → softsynth + soundfonts → mix/master → WAV/OGG (Opus)
   4b. Sheet     → NotationScore   → MusicXML → OSMD/VexFlow render service → SVG/PNG/PDF
   4c. Animation → PerformancePlan → note-event timeline → piano-roll video → **WebM (VP9 video + Opus audio)**
   Audio and animation consume the same PerformancePlan → they share one event schedule and are tested against the synchronization tolerance in §8.
   Notation is grounded in the NotationScore and never reflects humanization.
  │
  ▼
5. Job orchestration + web app
   - Long-running renders run as background jobs (see §7), not inline requests.
   - FastAPI web app: prompt box, job status, preview, downloads.
```

**v1 scope:** Western tonal music, **mood-based prompts only**. Famous-piece catalog, transforms, and Indian classical support are explicitly Phase 2/3. The v1 line in the previous draft that included famous-piece requests is removed.

## 3. LLM deployment strategy

**Dev/laptop environment (now):**
- Laptop can't host a local model at usable speed/quality, so development uses an **Ollama Cloud** hosted model selected by the Phase 1 benchmark in §10.
- The app talks to Ollama exclusively through the standard Ollama client/API (`base_url` + model identifier + API key) — **never hardcode "cloud" behavior into the app logic.**

**Production environment (later):**
- We will run our own self-hosted Ollama server (on infra we control) and point the app at it — same Ollama API shape, different `base_url` and an **environment-specific model identifier**.
- Because both dev and prod speak the identical Ollama API, switching is a **config change only** (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`), not a code change.

**Auth note:** Ollama Cloud requires a Bearer API key; self-hosted Ollama typically requires none (or an internal-network token). The client wrapper handles this transparently — the key is optional in config and only attached when present.

**Model-identifier note:** Cloud and self-hosted Ollama tag namespaces differ (`*-cloud` suffixes, availability, version pinning). We do **not** assume the same model name works in both environments. `OLLAMA_MODEL` is an environment-specific identifier; the chosen model is selected by benchmark (see §10, item 3).

**Structured-output capability note:** schema-enforced structured output has been verified against the actual Ollama Cloud endpoint configured for this project. Phase 1 sends the Pydantic-generated JSON Schema through Ollama's `format` option and validates the returned payload locally against the same Pydantic model. Native enforcement does not replace local validation. The `OllamaAdapter` performs a startup capability probe; if a different Ollama host does not support schema enforcement, it may fall back to an ordinary chat request with the schema and “JSON only” instruction embedded in the prompt, followed by the same repair/fallback policy in §6. The selected mode and probe result are recorded in job and benchmark metadata.

**Design implication:** the application depends on a small provider-neutral `LLMClient` interface. Phase 1 supplies one `OllamaAdapter`, configured by environment variables, which wraps the Ollama client. A future commercial provider would require its own adapter behind the same interface; provider-specific authentication and request mapping stay inside each adapter.

| Setting | Dev (now) | Production (later) |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama Cloud endpoint | Self-hosted Ollama server URL |
| `OLLAMA_MODEL` | Environment-specific cloud identifier | Environment-specific self-hosted identifier |
| `OLLAMA_API_KEY` | Required (Bearer) | Optional / not used |
| Auth header | `Authorization: Bearer <key>` | None (or internal-network token) |

## 4. Licensing policy

**Rule:** every dependency (library, engine, binary, sample/soundfont, font) must be under a permissive, commercial-use-friendly license — **Apache-2.0, MIT, BSD (2/3-clause), CC0, or an equivalent "free for commercial use, no source-disclosure obligation" license.** No copyleft (GPL/LGPL/AGPL) and no paid/proprietary-commercial-license software, unless explicitly re-approved later.

### Dependency license audit (current plan)

| Component | Role | License | OK under policy? |
|---|---|---|---|
| `music21` | score model, music theory | BSD-3-Clause | ✅ |
| `pydantic` | spec validation | MIT | ✅ |
| `FastAPI` | web backend | MIT | ✅ |
| `ollama` (server + python client) | LLM access | MIT | ✅ |
| `pydub` | audio mixing/mastering | MIT | ✅ |
| `moviepy` | animation/video rendering | MIT (Python wrapper) | ✅ **with caveat** — see FFmpeg note |
| `OpenSheetMusicDisplay` | **sheet music rendering** | BSD-3-Clause | ✅ |
| `VexFlow` (transitive via OSMD) | notation primitive | MIT | ✅ |
| `Playwright` | browser automation for render service | Apache-2.0 | ✅ |
| Chromium | headless browser runtime | BSD-style core + third-party components | ⚠️ notices/transitive audit required |
| FFmpeg | video/audio encode/decode (transitive via moviepy/pydub) | LGPL-2.1+ by default; GPL when GPL components are enabled; non-redistributable when configured `--enable-nonfree` | ⚠️ see note |
| `libvpx` | VP9 encoder used by FFmpeg | BSD-3-Clause + patent grant | ✅, pin/audit exact build |
| `libopus` | Opus encoder used by FFmpeg | BSD-style + royalty-free patent grants | ✅, pin/audit exact build |
| FluidSynth (external process) | MIDI → audio | LGPL-2.1 | ✅ accepted (see note) |
| Salamander Grand Piano | Phase 1 piano samples | CC BY 3.0 | ✅ with required attribution |
| `RQ` | background job execution | BSD-2-Clause | ✅ selected; use JSON serialization |
| Valkey | job queue transport | BSD-3-Clause | ✅ selected |
| Bravura | Phase 1 engraving glyphs | SIL OFL 1.1 | ✅ with bundled notices |
| Ollama benchmark winner | LLM service/model selected under §10 | per-model license + acceptable-use review | ⚠️ release gate |

### Flagged: FFmpeg (transitive via moviepy)

`moviepy` and `pydub` are permissively licensed wrappers, but they shell out to FFmpeg for encode/decode. FFmpeg is LGPL-2.1+ by default; enabling GPL components with `--enable-gpl` makes the resulting build GPL, while `--enable-nonfree` can make the resulting build non-redistributable. Action items:
- **Phase 1 reference target: macOS on Apple Silicon (`darwin-arm64`).** Build FFmpeg 8.1.2 from the official `ffmpeg-8.1.2.tar.xz` source archive (SHA-256 `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`) as a project-owned shared build. Required policy flags are `--disable-gpl --disable-nonfree --enable-shared --disable-static --enable-libvpx --enable-libopus`; the final build script must also pin and audit the `libvpx` and `libopus` inputs and record the complete configure output.
- **Additional targets:** use the same pinned FFmpeg release and codec set. The `lgpl-shared` variant from `BtbN/FFmpeg-Builds` is an acceptable candidate for Windows/Linux only after its source revision, dependency set, configure flags, and checksum pass the same audit. Adding an OS/architecture is a release gate, not an assumption that one binary is portable.
- Do **not** rely on a system FFmpeg whose build provenance is unknown — that puts us back under unknown terms.
- Apply the LGPL packaging checklist below to FFmpeg and FluidSynth; a notice alone is not sufficient.

The locally installed/system FFmpeg is not used unless its recorded configure flags pass this audit. This is a packaging concern, not a code-rewrite concern, but the reproducible build must pass before Phase 1 distribution begins.

### Resolved: video container and codecs (Phase 1)

The default moviepy MP4 path uses libx264 for video and AAC for audio. The BtbN LGPL FFmpeg builds **exclude libx264 and libx265** (GPL-encumbered codecs), so MoviePy's default, broadly compatible H.264/MP4 path is unavailable with that build. Phase 1 therefore uses a container/codec combination explicitly present in the audited LGPL build.

**Phase 1 decision: WebM with VP9 video + Opus audio.**
- VP9 and Opus are both royalty-free and available through permissively licensed `libvpx` and `libopus`; the selected FFmpeg build must prove that both were enabled without enabling GPL/non-free components.
- WebM is the standard royalty-free container for these codecs.
- This is a Phase 1 default; MP4 may return in a later phase if we ship a different FFmpeg build, accept GPL for the video stack, or move to a fully-permissive encoder (e.g. rav1e for AV1).
- The container, video codec, and audio codec are recorded in the artifact manifest (§9) and pinned at the version level — every release is reproducible only against a specific `ffmpeg <build-sha>` with a specific codec configuration.

**Phase 1 audio outputs:** uncompressed WAV plus OGG containing Opus audio. MP3 is deliberately excluded from Phase 1 so the dependency and codec policy is identical across audio-only and video artifacts. A later phase may add MP3 after its exact encoder build and license obligations are audited.

### Flagged: LGPL packaging compliance checklist

LGPL is permissive enough for this project, but "we invoke it as a subprocess" is only one part of compliance. For each LGPL component we ship (FluidSynth, FFmpeg, etc.) the distribution must include:

- **License text** for that component, in a `LICENSES/` directory of the distribution.
- **Source / build information** — for FFmpeg this means either shipping the exact source tree we built from (with build script) or a written offer to provide source on request; for pre-built binaries, document the upstream build provenance.
- **Modifications** — if we patch the LGPL component at all, the patches must be documented and the modified source made available.
- **Replacement rights** — LGPL specifically grants the user the right to replace the LGPL component with their own build. Our application invokes separate executables and distributes the shared-library FFmpeg variant; paths/configuration must remain replaceable and must not be obfuscated or locked down.
- **`--enable-gpl` / `--enable-nonfree` audit** — verify the FFmpeg build was *not* configured with `--enable-gpl` (which would make the resulting binary GPL, not LGPL) and *not* with `--enable-nonfree` (which can produce a non-redistributable binary). The pinned build's `./configure` flags must be in the manifest.

The artifact manifest (§9) carries the per-component build sha and a `license_obligations` block pointing at where each license text and source-offer lives in the distribution.

### Flagged: Playwright / Chromium distribution

Playwright is Apache-2.0, but the Chromium binary it installs is a separate runtime with a BSD-style core and many third-party components. Before bundling or downloading a pinned browser build for distribution:

- record the Chromium revision and download source;
- retain Chromium's bundled license and third-party notices;
- audit its shipped third-party component inventory against this policy; and
- record the browser revision and notice location in the artifact manifest/toolchain metadata.

The Playwright package license alone is not treated as approval for the complete browser binary.

### Flagged: notation font

Engraving fonts matter and were not in the previous audit. **Bravura**, the most common notation font (SMuFL standard reference font), is **SIL OFL 1.1** — not MIT/BSD/Apache. SIL OFL is permissive and commercial-friendly, but it is a font license, not a software license, and must be evaluated under the same permissive-policy lens:
- SIL OFL is acceptable under this policy. ✅ permitted license.
- Attribution and font-file redistribution obligations apply; document them in distribution notices.
- OSMD accepts arbitrary SMuFL-compliant fonts, so we are not locked to Bravura — but every candidate font must be license-checked at the version pinned.
- The render service must bundle the font file (or download it at install with a documented source) and the license file must ship with the distribution.

### Flagged: Ollama model weights

The Ollama server and Python client are MIT, but **Ollama does not license the model weights it serves.** Each candidate model on Ollama Cloud (and each model we'd self-host) carries its own license and acceptable-use policy — Gemma, DeepSeek, GLM, Llama, Mistral etc. each differ.

Required gate per model: **license + acceptable-use review before that model is allowed as `OLLAMA_MODEL`.** Record the model's license, source URL, version pin, and review date in the repository's [`MODELS.md`](../MODELS.md) registry. This is a per-model decision, not a one-time decision — switching the default model requires a new review.

### Resolved: job orchestration infrastructure

§7 introduces background-job execution, which is load-bearing infrastructure. The Phase 1 choice is **RQ + Valkey**:

- **Redis-the-server is not accepted under this policy.** Redis 7.4 introduced RSALv2/SSPL licensing, and later releases may offer other non-permissive or copyleft options; none meet this project's strict MIT/BSD/Apache-style requirement. Note this is about the server software, not the protocol — `redis://`-style URLs are fine because **Valkey** speaks the same protocol wire-compatibly.
- **Valkey** is BSD-3-Clause and wire-compatible with the Redis protocol. Phase 1 pins a Valkey release and uses it only on the local interface.
- **RQ** is BSD-2-Clause, actively maintained, and a better lifecycle choice than `arq`, which is in maintenance-only mode. RQ's default pickle serializer must not be used; enqueue only primitive job arguments through `JSONSerializer` so broker contents cannot trigger arbitrary-code deserialization.
- Worker jobs receive only a `job_id` and load the canonical prompt/spec from application storage. Render subprocesses run with explicit timeouts and bounded output paths.

### Resolved: notation rendering

The previous draft named bare VexFlow; VexFlow is a low-level layout primitive and does not consume MusicXML directly. **OpenSheetMusicDisplay (OSMD)** is the explicit MusicXML → VexFlow bridge, BSD-3-Clause, and supports both browser and headless rendering. The render path is therefore:

`NotationScore → MusicXML → OpenSheetMusicDisplay/VexFlow render service (headless Chromium via Playwright) → SVG / PNG / PDF`

This adds a render-service component (a small Node + headless-Chromium process) to the stack, which is the cost of staying copyleft-free without rolling our own MusicXML layout.

### Resolved: audio synthesis engine

**FluidSynth accepted as an external LGPL-2.1 process.** LGPL permits commercial/proprietary use when the library is invoked as an external process or dynamically linked, with no obligation to open-source our own application code. FluidSynth is invoked as a separate process (not linked into our code), satisfying the standard packaging-compliance posture. Building a custom sampler is unnecessary MVP risk and is deferred (if ever needed) behind this decision.

### Flagged: soundfonts / sample libraries

**Previous draft was wrong on two counts and is corrected here.** Per-asset license check is required before use; some widely-used GM soundfonts have informal or unclear licensing. Candidates evaluated:

- **Salamander Grand Piano** — **CC BY 3.0** (attribution required), **not CC0/public domain**. Distribution must include, in a credits/notice file:
  - Author: Alexander Holm
  - Source: https://salamanderan.com/ (and/or the specific source URL of the downloaded release)
  - License: CC BY 3.0 — https://creativecommons.org/licenses/by/3.0/
  - Modification status: **unmodified** for Phase 1 (we ship the original sample pack as-is). If we ever convert the format, retune, or otherwise modify it, that fact must be stated explicitly per CC BY 3.0 §3(b).
  Acceptable under permissive policy, but conditional on correct attribution handling. ✅ conditional.
- **GeneralUser GS** — **DO NOT USE** under a strict licensing policy. Its own license explicitly disclaims provenance: the author states they cannot guarantee the origin of every sample. That uncertainty breaks the "we know what we ship and under what terms" assumption this policy is built on. Off the candidate list until/unless we find a fully-traceable GM alternative.
- **Orchestral / other instrument samples** (needed for non-piano instruments, and later for Indian classical instruments) — will need individual sourcing + license verification in Phase 2/3; nothing committed here yet. The same provenance-traceability rule that disqualified GeneralUser GS applies to every candidate going forward: every sample must trace to a known license.

**Net Phase 1 soundfont plan:** Salamander Grand Piano (CC BY 3.0, attribution included) for piano; everything else deferred to Phase 2 once we've sourced license-clean alternatives.

**Action item:** none of the above license facts should be treated as final legal sign-off — confirm current license text for each asset/library at the version you actually pin, since licenses can change between releases.

## 5. Phased roadmap

**Phase 1 — MVP (deliberately small — validate the full pipeline before expanding scope)**
- **Piano only** — one instrument, one soundfont (Salamander Grand Piano, CC BY 3.0).
- **Three moods** — `calming | electrifying | sleep`, no others.
- **Fixed forms** — a small bounded set (e.g. 8/16/32-bar forms with predefined harmonic templates per mood); no free-form composition in v1. This is the constraint that keeps the rule-based engine tractable and the acceptance tests deterministic.
- **Duration policy: target, arrange, then fine-tune.** The spec's `duration_seconds` is a target. The engine selects a fixed form and mood-appropriate tempo range, repeats or varies complete sections, and may add a shorter coda made only of complete measures. It then fine-tunes tempo within the mood's allowed range to reach the target within the ±2% tolerance from §8. It never truncates a sounding note or emits an incomplete measure. If no valid arrangement can satisfy the tolerance, composition fails with a structured `duration_unfulfillable` error rather than silently returning the wrong duration. `PerformancePlan.realized_duration_seconds` records the result. No source section may repeat more than eight times; repeated sections receive deterministic, seed-derived variation in at least one of accompaniment voicing, register, dynamics, or rhythm.
- **No known-piece catalog.** Famous-piece requests are out of v1 entirely; users cannot ask for Canon-in-D-style output in Phase 1.
- **No creative-assist LLM.** Composition is fully rule-based in Phase 1; the LLM is used only for prompt parsing, not for melodic/harmonic seeds.
- Ollama Cloud schema-enforced structured output for parsing + spec extraction, with local schema validation, repair, and deterministic fallback. **Per-model license + acceptable-use review completed before that model is allowed as `OLLAMA_MODEL`** (see §4).
- `CompositionSpec` schema defined up front (see §6) — bounded to the Phase 1 vocabulary, with seed + engine version for reproducibility.
- Rule-based composition engine (`music21`), producing NotationScore + PerformancePlan.
- Audio via FluidSynth (external process) + Salamander Grand Piano (CC BY 3.0, attribution included).
- Sheet music via OSMD/VexFlow render service + SMuFL notation font (license-checked, e.g. Bravura SIL OFL with attribution).
- Piano-roll animation from PerformancePlan.
- FFmpeg 8.1.2 built as the project-owned LGPL shared build for the Phase 1 `darwin-arm64` reference target; additional targets must pass the same build/license gate (see §4).
- RQ worker + Valkey broker, using RQ `JSONSerializer`; exact releases are pinned in the implementation lockfiles (see §4).
- Artifact manifest emitted with every completed job (see §9).
- Local FastAPI web app: prompt in, job status, preview, downloads.
- Job model + acceptance criteria implemented from day one (see §7, §8).

**Phase 2**
- Known-piece catalog + transforms (Canon in D, etc.) — **the v1 contradiction with Phase 2 is resolved here**. Each catalog entry must carry **both** a composition-copyright column (is the underlying composition in the public domain in our target jurisdictions?) **and** an edition/file-licensing column (under what terms is the specific MusicXML/MIDI file we're shipping?). Both must be permissive per §4. Public-domain composition + non-permissive file = still fails the audit. Catalog entries without both columns filled in are not shippable.
- Optional cloud-LLM creative assist for melody/harmony seeds (still Ollama-routed).
- Expanded instrumentation (each new instrument needs a license-clean sample, see §4 soundfont rules), humanization, mixing polish.
- Config-driven swap from Ollama Cloud to self-hosted Ollama server.
- Acceptance criteria expanded to cover humanization invariants.

**Phase 3**
- Indian classical module: raga scale/ornamentation rules, tala rhythmic cycles, license-checked sitar/tabla/bansuri samples. Treated as its own design effort, not an instrument re-skin of the Western engine.

**Phase 4 (optional)**
- Investigate a license-clean neural audio-generation model as a second, clearly-labeled "produced" audio render, sheet music still sourced from the NotationScore.

## 6. CompositionSpec schema (defined now, used in Phase 1)

A versioned Pydantic schema is the contract between the prompt parser and every downstream component. Its Phase 1 version is defined up front and frozen before the first Phase 1 release; later breaking changes require a new `schema_version`.

**Required fields (Phase 1 subset):**
- `schema_version: int` — incremented on any breaking change.
- `request_kind: enum` — `mood_generation` (only Phase 1 value); `famous_piece` is reserved.
- `duration_seconds: int` — bounded (30 ≤ x ≤ 600), default 180. The engine reaches this target using the Phase 1 target/arrange/fine-tune policy and reports the realized duration.
- `tempo_bpm: int | None` — bounded range (e.g. 40 ≤ 240); `None` means the engine derives it from the mood template and reports the chosen value.
- `key: enum | None` — bounded to Western keys; `None` means the engine chooses.
- `time_signature: enum` — bounded to common signatures.
- `mood: enum` — bounded Phase 1 vocabulary: `calming | electrifying | sleep`.
- `instrumentation: Literal["piano"]` — Phase 1 is piano-only; the field is a single literal, not a list. Phase 2+ will widen this to `list[enum]`.
- `seed: int | None` — for reproducibility; `None` means the engine chooses and reports.
- `humanization: Literal["none"]` — only `none` is allowed in Phase 1.

**Behavior on unsupported requests:**
- Out-of-vocabulary values → rejected by the parser with a structured error, not silently coerced.
- Ollama failure / rate-limit / non-conforming JSON after 2 repairs → deterministic fallback parser for the bounded mood subset; if fallback also fails, the job fails with a clear error message and the prompt returned to the user for editing.

**Repair/retry policy:**
- LLM output is validated against the schema before acceptance. Invalid JSON or schema-invalid output triggers up to 2 repairs (re-prompt with schema-in-context error feedback). After that, fallback parser is used for in-vocabulary requests; out-of-vocabulary requests fail.
- Every accepted spec is persisted with the parser source (`llm` | `fallback`) and a hash, for reproducibility audits.

### Canonical symbolic serialization

`CompositionSpec`, `NotationScore`, and `PerformancePlan` are persisted as canonical JSON documents, not hashed from in-memory Python or `music21` objects. Canonicalization rules are part of the format contract:

- UTF-8 JSON, sorted object keys, no insignificant whitespace, and no NaN/Infinity values;
- integer musical ticks for score positions and durations;
- integer microseconds for realized performance timestamps;
- integer MIDI pitches and velocities;
- arrays remain in musically significant order, with deterministic secondary sorting where events share a timestamp; and
- an explicit format version in each document.

Hashes are calculated over these canonical UTF-8 bytes. MusicXML, MIDI, PDF, audio, and video are derived renderer outputs and are not the canonical symbolic representation.

## 7. Job model (long-running renders)

Five minutes of audio, notation, and video is too long for a single FastAPI request. Renders run as background jobs.

**State machine:**
```
queued → parsing → composing → validating → rendering_audio → rendering_sheet → rendering_animation → complete
          │                         │                                                        │
          └─────────────────────────┴──────────────── failed ────────────────────────────────┘
```

**Failed is reachable from every state** (the diagram abbreviates the arrows for readability). Implementers must treat `failed` as a terminal sink for any non-terminal state.

**Status fields per job:**
- `job_id`, `created_at`, `updated_at`
- `state`, `progress` (0.0–1.0 per stage), `current_stage`
- `input_prompt` (the original user prompt)
- `input_spec` (the persisted CompositionSpec; null until parsing succeeds)
- `artifacts`: `{audio: {path, sha256}, sheet: {path, sha256}, animation: {path, sha256}}`
- `error`: structured error with stage + reason
- `engine_version`, `seed` — for reproducibility

**Authentication / authorization:**
- Phase 1 is **single-user / local-only.** The FastAPI app runs on the user's own machine, jobs are scoped to that machine, and there is no concept of a job owner or per-user auth.
- Artifact URLs are opaque, unguessable tokens scoped to the job ID. That is sufficient for a single-user local deployment.
- Multi-user auth, per-user job scoping, and the "owner-scoped" API surface are explicitly **deferred to Phase 2+.** When they're added, the API surface changes and the owner field is reintroduced.

**Retention:**
- Completed artifacts are retained for 7 days, then pruned.
- Failed jobs are retained for 24 hours for debugging.

**API surface:**
- `POST /jobs` — accepts `{prompt: string}`, creates the job, and returns `job_id`. Prompt parsing occurs in the worker's `parsing` stage.
- `POST /jobs/from-spec` — internal/test-only Phase 1 endpoint that accepts an already validated `CompositionSpec`; it is not exposed by the user-facing UI.
- `GET /jobs/{id}` — returns state + progress + artifact URLs once complete.
- `GET /jobs/{id}/artifact/{kind}` — streams the artifact (opaque, unguessable token tied to the job ID; no auth needed in Phase 1 since the app is single-user/local).

**Cancel:** `POST /jobs/{id}/cancel` — allowed in `queued`, `parsing`, and `composing`. It returns `409 Conflict` once `rendering_*` has started because Phase 1 does not support mid-render teardown.

## 8. MVP acceptance criteria

*(Acceptance criteria are defined before implementation begins — they are the bar for "MVP done," not the spec for any single feature.)*

Phase 1 is "done" only when every criterion below is met, measured by an automated test suite run against every release.

**Spec & reproducibility:**
- Canonically serialized `CompositionSpec`, `NotationScore`, and `PerformancePlan` are **byte-identical** given the same spec + seed + engine version + pinned composition dependencies. These are the canonical, hashed artifacts; their serialization rules are defined in §6.
- **Media artifacts (WAV/OGG, SVG/PNG/PDF, WebM) are not required to be byte-identical across runs, even on the same toolchain.** Encoders can include container metadata, timestamps, platform-specific font output, or non-deterministic muxing. Reproducibility for media is checked **semantically**: same set of scheduled note events, same durations within tolerance, same dimensions, and same audio levels within tolerance. The manifest's `toolchain` block records exactly how a specific media file was produced; its artifact hash verifies that stored file's integrity, not a promise that a future render will have the same hash.
- Every accepted spec round-trips through the schema validator.

**Composition correctness:**
- All notes within instrument range.
- All measures complete (no dropped beats).
- No unresolved voice-leading collisions flagged by the theory linter.
- Generated MusicXML validates against the MusicXML schema.
- Generated MIDI is well-formed (parseable by `mido`/`music21`).

**Duration tolerance:**
- Final audio duration within ±2% of `duration_seconds` in the spec.
- `PerformancePlan.realized_duration_seconds` agrees with decoded audio duration within the documented encoder-delay tolerance.

**Render performance:**
- A 5-minute piece renders end-to-end in ≤ 15 minutes (≤ 3× real-time) on the recorded Phase 1 reference machine, with no individual stage stalled for more than 5 minutes without progress.

**Audio / animation synchronization:**
- Scheduled MIDI note-on events and piano-roll note-on events use the same integer-microsecond timestamps from `PerformancePlan`.
- End-to-end tests decode the rendered audio/video and account for documented SoundFont attack and Opus encoder pre-skip; measured audible/visible onset agrees within ±20 ms after those declared offsets are applied.

**Notation correctness:**
- OSMD render produces a valid SVG/PNG/PDF with no rendering errors.
- Notation reflects NotationScore (not PerformancePlan); measures match the spec's time signature.

**Parser robustness:**
- The benchmark contains exactly 100 version-controlled, hand-labeled prompts at `tests/fixtures/parser_benchmark.jsonl`: in-scope paraphrases, boundary durations, malformed requests, and unsupported requests.
- [`docs/parser-benchmark.md`](parser-benchmark.md) defines the corpus schema, category balance, labeling rules, ambiguity policy, review process, and benchmark command. Each expected result receives a second human review before the corpus is frozen; disagreements are adjudicated and recorded. Pre-adjudication field-level agreement must be at least 90%.
- First responses are valid JSON matching `CompositionSpec` at least 98% of the time; after at most two repairs or deterministic fallback, all supported benchmark prompts parse successfully.
- Field-level semantic exact-match accuracy is at least 97% across supported prompts, and unsupported-request rejection accuracy is at least 95%.
- Cloud parsing p95 latency is ≤ 5 seconds on the recorded reference connection. A model that misses any quality threshold is disqualified regardless of speed or cost.

## 9. Artifact manifest (every completed job)

Every completed job emits a machine-readable `manifest.json` alongside the artifacts. It is the reproducibility and provenance record for that specific render.

**Required fields:**
- `job_id`, `created_at`, `completed_at`
- `input_spec`: the full `CompositionSpec` (schema_version included) and its sha256
- `seed`, `engine_version`
- `parser_source`: `llm` | `fallback`, plus the model identifier when `llm`
- `notation_score_sha256`, `performance_plan_sha256`
- `artifacts`: per-artifact `{kind, container, codec, path, sha256, size_bytes}`
- `toolchain`: per-artifact `{engine, version, build_sha, config}` — e.g. `fluidsynth 2.x`, `osmd 1.x`, `chromium <revision>`, `ffmpeg <build-sha>` configured with `--enable-libvpx --enable-libopus`
- `assets`: `{soundfont: {name, version, sha256, license, source_url}, notation_font: {…}}`
- `dependencies`: pinned versions of every library involved in the render
- `license_obligations`: pointers into the distribution's `LICENSES/` directory for every LGPL/LGPL-conditional component, plus the FFmpeg `--configure` flag set used to build the bundled FFmpeg.

The manifest is what makes the canonical artifacts (spec, scores, plans) reproducible hash-for-hash and what makes license audits tractable post-hoc — every shipped artifact points at every asset and version that produced it. For media artifacts, the manifest's `toolchain` block records provenance (encoder version, build sha, browser revision, container, and codec configuration); media are verified semantically per §8 rather than expected to reproduce hash-for-hash.

## 10. Resolved Phase 1 decisions and release gates

There are no remaining architecture choices blocking implementation. The decisions below are approved; items described as gates produce evidence during implementation and CI rather than requiring another design decision.

1. **Audio synthesis:** FluidSynth as a replaceable external LGPL process, subject to the §4 packaging checklist.
2. **Notation rendering:** OpenSheetMusicDisplay/VexFlow in Playwright-managed headless Chromium, with Chromium's complete third-party notice audit retained.
3. **Ollama model selection:** do not hardcode a model by preference. [`MODELS.md`](../MODELS.md) must exist before the first candidate-model request is made. Before the Phase 1 model is pinned, query the configured Ollama Cloud host for currently available text models, discard any model that fails the license/acceptable-use gate, and run the 100-prompt benchmark in §8. The least expensive model meeting every quality and latency threshold becomes `OLLAMA_MODEL`; ties are broken by lower p95 latency. Record the model identifier, service date, benchmark results, license, terms URL, review date, and structured-output capability result in `MODELS.md`. The configured Phase 1 Cloud endpoint has been verified to support native schema-enforced structured output, so that is the benchmark and production default; prompted JSON is only the compatibility fallback for another host that fails the capability probe.
4. **Benchmark corpus:** `docs/parser-benchmark.md` and the reviewed `tests/fixtures/parser_benchmark.jsonl` corpus are release-gate artifacts. The labeling guide is approved before prompts are labeled; the corpus is frozen and versioned before candidate models are scored. Candidate-model outputs must never be used to define or revise the expected labels during the same selection run.
5. **Acceptance thresholds:** duration ±2%; audio/animation onset ±20 ms after declared renderer offsets; five-minute end-to-end render ≤15 minutes on the recorded reference machine; parser thresholds as specified in §8.
6. **Retention:** completed jobs and artifacts for 7 days; failed jobs for 24 hours. Retention is configurable, and cleanup must never delete an active job.
7. **Notation font:** Bravura under SIL OFL 1.1, bundled with its copyright and license notices.
8. **FFmpeg:** official FFmpeg 8.1.2 source archive, SHA-256 `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`, built as an LGPL shared build for the Phase 1 `darwin-arm64` target using the policy flags in §4. The build script, complete configure output, dependent-library pins, and resulting binary hashes are release artifacts. The installed Homebrew/system FFmpeg is not used unless it independently passes the same audit.
9. **Jobs:** RQ + Valkey, bound locally, with RQ `JSONSerializer` and primitive job arguments. Exact versions are pinned in lockfiles; an integration test against the selected Valkey release is a release gate.
10. **Video:** WebM with VP9 through `libvpx` and Opus through `libopus`. An implementation spike must meet the §8 quality/performance criteria; failure triggers a documented architecture amendment and new codec license audit rather than an implicit fallback.
11. **Repetition:** maximum eight repetitions of any source section, deterministic seeded variation on every repeat, requested duration capped at 600 seconds, and realized duration constrained to §8.
12. **Salamander attribution:** canonical notice at `LICENSES/Salamander-Grand-Piano.txt`, referenced by `THIRD_PARTY_NOTICES.md`, exposed in the UI credits view, and included in downloadable bundles containing rendered Salamander audio. Where supported, standalone audio metadata also records the library, author, and license URL.
