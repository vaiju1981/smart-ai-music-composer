# saimc-render-service

Sheet-music renderer for the smart-ai-music-composer pipeline:
MusicXML in, SVG (or PNG) out, via
[OpenSheetMusicDisplay](https://github.com/opensheetmusicdisplay/opensheetmusicdisplay)
in Playwright-managed headless Chromium.

The Python worker invokes this as a one-shot CLI per job — no
long-running service process.

## Install

Node 18 is the pinned toolchain (`engines: >=18`); install with nvm so
npm uses it:

```sh
nvm use 18
npm install
npm run build                      # tsc -> dist/
npx playwright install chromium    # downloads the pinned headless browser
```

## Usage

```sh
node dist/cli.js render --input score.musicxml --output sheet.svg
node dist/cli.js render --input score.musicxml --output sheet.png --format png
```

Exit codes: `0` success, `2` usage error, `3` render failure (missing
input, non-MusicXML input, Chromium unavailable, OSMD page error).

## How it works

- `src/cli.ts` — argument parsing + exit codes
- `src/render.ts` — reads/validates the MusicXML, launches Chromium,
  loads `src/viewer/viewer.html`, and calls the page's
  `renderMusicXml(xml)` bridge
- `src/viewer/viewer.js` — browser-side OSMD harness; the SVG markup is
  taken from the container, or a PNG via element screenshot
- No bundler: `viewer.html` loads OSMD's prebuilt UMD bundle directly
  from this repo's `node_modules`

## Tests

```sh
npm test                # builds, then unit tests (no Chromium required)
npm run test:integration  # real Chromium + real OSMD; skips when the browser is absent
npm run typecheck       # tsc --noEmit
```

## Licensing

Third-party notices live in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) with full license
texts under [`licenses/`](licenses/): OpenSheetMusicDisplay (BSD-3),
VexFlow (MIT, bundles the Bravura/SMuFL glyph data under SIL OFL), and
Playwright (Apache-2.0). The Playwright-managed Chromium browser binary
carries its own BSD-style license plus third-party components; its
complete notice set is part of the Phase 1 release-gate audit
(docs/roadmap.md §10 #2).