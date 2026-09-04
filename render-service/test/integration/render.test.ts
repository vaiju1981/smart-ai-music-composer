/**
 * Integration tests: real Chromium + real OSMD.
 *
 * These need the Playwright Chromium download (`npx playwright install
 * chromium`) and are skipped automatically when the browser binary is
 * absent, so the default test run stays green in environments without
 * it.
 */

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { chromium } from "playwright";
import { renderSheet } from "../../src/render.js";

const browserInstalled = (() => {
  try {
    return existsSync(chromium.executablePath());
  } catch {
    return false;
  }
})();

/** Minimal valid MusicXML 4.0 document: one part, 4/4, one whole-note C4. */
const SAMPLE_MUSICXML = `<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>`;

describe("renderSheet end-to-end", { skip: !browserInstalled }, () => {
  it("renders SVG markup", async () => {
    const dir = await mkdtemp(join(tmpdir(), "saimc-render-e2e-"));
    const input = join(dir, "sample.musicxml");
    const output = join(dir, "sheet.svg");
    await writeFile(input, SAMPLE_MUSICXML, "utf8");

    await renderSheet({ inputPath: input, outputPath: output, format: "svg" });

    const svg = readFileSync(output, "utf8");
    assert.ok(svg.includes("<svg"), "output should contain an <svg> element");
  });

  it("renders a PNG screenshot", async () => {
    const dir = await mkdtemp(join(tmpdir(), "saimc-render-e2e-"));
    const input = join(dir, "sample.musicxml");
    const output = join(dir, "sheet.png");
    await writeFile(input, SAMPLE_MUSICXML, "utf8");

    await renderSheet({ inputPath: input, outputPath: output, format: "png" });

    const png = readFileSync(output);
    // PNG magic bytes.
    assert.deepEqual(
      [...png.subarray(0, 8)],
      [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    );
  });
});