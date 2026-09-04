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