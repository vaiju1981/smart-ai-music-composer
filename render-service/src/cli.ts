#!/usr/bin/env node
/**
 * CLI entry point for the sheet render service.
 *
 * One-shot invocation model: the Python worker calls
 * `node render-service/src/cli.ts render --input x.musicxml --output y.svg`
 * per job through its subprocess helper, so no long-running service
 * process has to be managed.
 *
 * Exit codes: 0 success, 2 usage error, 3 render failure.
 */

import { pathToFileURL } from "node:url";
import { RenderError, renderSheet, type SheetFormat } from "./render.js";

const USAGE = `usage: cli.ts render --input <musicxml> --output <file> [--format svg|png]

render one MusicXML document with OpenSheetMusicDisplay in headless
Chromium and write the sheet artifact (svg markup or png screenshot).`;

export class CliUsageError extends Error {}

export interface ParsedArgs {
  input: string;
  output: string;
  format: SheetFormat;
}

const FORMATS: readonly SheetFormat[] = ["svg", "png"];

export function parseArgs(argv: readonly string[]): ParsedArgs {
  if (argv.length === 0 || argv[0] !== "render") {
    throw new CliUsageError(`expected command "render"\n${USAGE}`);
  }

  let input: string | undefined;
  let output: string | undefined;
  let format: SheetFormat = "svg";

  for (let i = 1; i < argv.length; i += 2) {
    const flag = argv[i];
    const value = argv[i + 1];
    switch (flag) {
      case "--input":
      case "--output": {
        if (value === undefined) {
          throw new CliUsageError(`missing value for ${flag}\n${USAGE}`);
        }
        if (flag === "--input") input = value;
        else output = value;
        break;
      }
      case "--format": {
        if (value !== "svg" && value !== "png") {
          throw new CliUsageError(
            `unsupported format ${String(value)} (expected svg|png)`,
          );
        }
        format = value;
        break;
      }
      default:
        throw new CliUsageError(`unknown flag ${String(flag)}\n${USAGE}`);
    }
  }

  if (input === undefined || output === undefined) {
    throw new CliUsageError(`--input and --output are required\n${USAGE}`);
  }
  return { input, output, format };
}

export async function main(argv: readonly string[]): Promise<number> {
  let args: ParsedArgs;
  try {
    args = parseArgs(argv);
  } catch (err) {
    if (err instanceof CliUsageError) {
      process.stderr.write(`${err.message}\n`);
      return 2;
    }
    throw err;
  }

  try {
    await renderSheet({
      inputPath: args.input,
      outputPath: args.output,
      format: args.format,
    });
  } catch (err) {
    if (err instanceof RenderError) {
      process.stderr.write(`render failed: ${err.message}\n`);
      return 3;
    }
    throw err;
  }
  process.stdout.write(`wrote ${args.output}\n`);
  return 0;
}

function isDirectInvocation(moduleUrl: string): boolean {
  const entry = process.argv[1];
  return entry !== undefined && pathToFileURL(entry).href === moduleUrl;
}

if (isDirectInvocation(import.meta.url)) {
  process.exitCode = await main(process.argv.slice(2));
}