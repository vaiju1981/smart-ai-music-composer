/** Unit tests for CLI argument parsing (no Chromium involved). */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { CliUsageError, parseArgs } from "../src/cli.js";

describe("parseArgs", () => {
  it("parses render with input/output and defaults format to svg", () => {
    const args = parseArgs(["render", "--input", "a.musicxml", "--output", "b.svg"]);
    assert.deepEqual(args, { input: "a.musicxml", output: "b.svg", format: "svg" });
  });

  it("accepts --format png", () => {
    const args = parseArgs([
      "render",
      "--input",
      "a.musicxml",
      "--output",
      "b.png",
      "--format",
      "png",
    ]);
    assert.equal(args.format, "png");
  });

  it("rejects a missing or wrong command", () => {
    assert.throws(() => parseArgs([]), CliUsageError);
    assert.throws(() => parseArgs(["renderx"]), CliUsageError);
  });

  it("rejects unknown flags", () => {
    assert.throws(
      () => parseArgs(["render", "--wat", "x"]),
      CliUsageError,
    );
  });

  it("rejects missing flag values", () => {
    assert.throws(
      () => parseArgs(["render", "--input"]),
      CliUsageError,
    );
  });

  it("rejects unsupported formats", () => {
    assert.throws(
      () =>
        parseArgs([
          "render",
          "--input",
          "a.musicxml",
          "--output",
          "b.pdf",
          "--format",
          "pdf",
        ]),
      /unsupported format/,
    );
  });

  it("rejects missing --input or --output", () => {
    assert.throws(() => parseArgs(["render", "--output", "b.svg"]), /required/);
    assert.throws(() => parseArgs(["render", "--input", "a.musicxml"]), /required/);
  });
});