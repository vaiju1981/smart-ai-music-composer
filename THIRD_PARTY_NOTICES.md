# Third-party notices

Components bundled with or invoked by Smart AI Music Composer, and the
terms under which each is distributed or used. The FFmpeg audit gate
(`src/saimc/render/ffmpeg_audit.py`, roadmap §10 #7) enforces the
LGPL posture described here at every render.

## FFmpeg

- License: LGPL-2.1-or-later as invoked and distributed here. The
  project builds FFmpeg with `--enable-libvpx --enable-libopus` and
  **without** `--enable-gpl` or `--enable-nonfree`
  (`scripts/build_ffmpeg.sh`); every binary is audited before use and
  GPL/nonfree builds are rejected. The exact `configuration:` line of
  the binary that rendered a given job is recorded in that job's
  `manifest.json`.
- Build script and pinned release: `scripts/build_ffmpeg.sh`.
- The LGPL notices for the build are bundled with the binary's
  distribution archive; FFmpeg's source is available from
  https://ffmpeg.org/.

## FluidSynth

- License: LGPL-2.1-or-later. Invoked as an external binary to render
  MIDI to WAV; the exact version used is recorded in each job's
  `manifest.json` (`toolchain.audio`).
- Source: https://www.fluidsynth.org/

## Salamander Grand Piano (soundfont)

- License: CC BY 3.0 — attribution required.
- Author: Alexander Holm; source:
  https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
  (original recordings: https://archive.org/details/SalamanderGrandPianoV3)
- Canonical notice: [`LICENSES/Salamander-Grand-Piano.txt`](LICENSES/Salamander-Grand-Piano.txt)
- Rendered audio carries the attribution in embedded metadata, and
  each completed job's `manifest.json` records the soundfont's
  version, sha256, and this notice path.

## FluidR3 General MIDI (soundfont)

- License: MIT — commercial use permitted, no attribution required
  (the author requests credit if used to derive new public soundfont
  banks).
- Author: Frank Wen, version 3.1. Covers every instrument in
  `INSTRUMENT_PROGRAMS` that Salamander does not render, including the
  non-western voices GM provides (sitar, koto, shanai, taiko,
  kalimba).
- Downloaded with a pinned sha256 by `scripts/download_soundfonts.py`
  from the Debian `fluid-soundfont` package (upstream orig tarball).
- The license text ships alongside the font
  (`assets/soundfonts/FluidR3_COPYING.txt`) and each completed job's
  `manifest.json` records the soundfont's sha256.

## 105-Sitar (soundfont)

- License: Public domain (dedicated to the public domain by the
  uploader) — unrestricted use including commercial.
- Source: Musical Artifacts artifact #3847
  (https://musical-artifacts.com/artifacts/3847), file
  `assets/soundfonts/105-Sitar.sf2`, sha256
  `5a7941e74d9a7f8c5bbc18a68b4f5d27cff540d591b1359b8245e8c561a9df93`
  (pinned in `scripts/download_soundfonts.py`; the pin is verified by
  `--status`).
- Renders the `sitar` instrument when present (falling back to
  FluidR3's GM sitar otherwise); single preset at bank 0, program 0,
  selected via `FONT_PRESETS`.

## Wetthasinghe's Harmonium (soundfont)

- License: CC-BY 4.0 — commercial use permitted with attribution.
- Author: Wetthasinghe. Source: Musical Artifacts artifact #1391
  (https://musical-artifacts.com/artifacts/1391), file
  `assets/soundfonts/Wetthasinghe_Harmonium.sf2`, sha256
  `ea2e31c26057dd9c39d08b2fd060e7b2f41e84a10aa1c61cec43d888a3ef0002`
  (pinned in `scripts/download_soundfonts.py`; the pin is verified by
  `--status`).
- Renders the `harmonium` instrument (General MIDI has no harmonium
  voice); single preset at bank 0, program 0, selected via
  `FONT_PRESETS`. CC-BY attribution is carried by this file and the
  job manifest's soundfont record.

## OpenSheetMusicDisplay (render-service)

- License: BSD-3-Clause, version 2.1.2, pinned in
  `render-service/package.json`.
- Notices: `render-service/THIRD-PARTY-NOTICES.md` (covers VexFlow
  and the Bravura SMuFL glyphs, SIL OFL 1.1).

## Playwright (render-service)

- License: Apache-2.0, version 1.61.0, pinned in
  `render-service/package.json`.
- Notices: `render-service/THIRD-PARTY-NOTICES.md`.

## Python libraries

Pinned versions and their licenses are recorded per job in
`manifest.json` (`dependencies`); distribution terms for each are in
its upstream package. Key render-path libraries:

- music21 (MusicXML export): MIT
- mido (MIDI): MIT
- Pillow (animation frames): MIT-CMU (HPND)