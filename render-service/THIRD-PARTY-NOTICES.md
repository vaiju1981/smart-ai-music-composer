# Third-party notices — saimc-render-service

Full license texts for the pinned npm dependencies live under
[`licenses/`](licenses/). Versions are pinned exactly in
[`package.json`](package.json).

| Component | Version | License | Source |
| --- | --- | --- | --- |
| OpenSheetMusicDisplay | 2.1.2 | BSD-3-Clause | https://github.com/opensheetmusicdisplay/opensheetmusicdisplay |
| VexFlow (OSMD dependency) | 1.2.93 | MIT | https://github.com/0xfe/vexflow |
| Playwright | 1.61.0 | Apache-2.0 | https://github.com/microsoft/playwright |
| Chromium (Playwright-managed browser) | Playwright-pinned revision | BSD-style + third-party components | https://www.chromium.org |

**Notation glyphs.** VexFlow bundles SMuFL notation glyph data,
including Bravura (SIL Open Font License 1.1). Per roadmap §10 #7, the
Bravura copyright and license notices travel with every distribution of
rendered notation output; see `licenses/` for the bundled font notice
and the repo root `LICENSES/` directory for asset attributions.

**Chromium.** The browser binary is downloaded at install time, not
committed to this repository. Its complete third-party notice audit is
retained as a Phase 1 release-gate artifact (docs/roadmap.md §4, §10 #2);
the audit covers the Chromium build for the pinned Playwright revision.