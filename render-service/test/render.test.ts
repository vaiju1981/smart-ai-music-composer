/** Unit tests for renderSheet input validation (no Chromium involved). */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtemp, writeFile } from "node:fs/promises";
import { RenderError, renderSheet } from "../src/render.js";

describe("renderSheet input validation", () => {
  it("rejects a missing input file before touching the browser", async () => {
    const dir = await mkdtemp(join(tmpdir(), "saimc-render-"));
    await assert.rejects(
      renderSheet({
        inputPath: join(dir, "missing.musicxml"),
        outputPath: join(dir, "out.svg"),
        format: "svg",
      }),
      RenderError,
    );
  });

  it("rejects input that is not MusicXML", async () => {
    const dir = await mkdtemp(join(tmpdir(), "saimc-render-"));
    const input = join(dir, "not-musicxml.xml");
    await writeFile(input, "<html><body>nope</body></html>", "utf8");
    await assert.rejects(
      renderSheet({ inputPath: input, outputPath: join(dir, "out.svg"), format: "svg" }),
      /not MusicXML/,
    );
  });
});