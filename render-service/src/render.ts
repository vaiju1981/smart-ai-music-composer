/**
 * Playwright + OSMD orchestration.
 *
 * Launches headless Chromium, loads the OSMD harness page, renders one
 * MusicXML document, and writes the sheet artifact:
 *
 *   svg -> the container's SVG markup, written as UTF-8 text
 *   png -> an element screenshot of the rendered page
 *
 * No browser is launched until the input has been read and sanity
 * checked, so argument/IO errors are cheap and unit-testable.
 */

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { chromium } from "playwright";

export type SheetFormat = "svg" | "png";

export class RenderError extends Error {}

export interface RenderSheetOptions {
  inputPath: string;
  outputPath: string;
  format: SheetFormat;
}

declare global {
  interface Window {
    renderMusicXml: (xml: string) => Promise<string>;
  }
}

// import.meta.dirname needs Node >=20.11; resolve from the module URL
// so the same code runs on the pinned Node 18 toolchain.
const VIEWER_URL = pathToFileURL(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "viewer", "viewer.html"),
).href;

/** Cheap structural check so a wrong-format input fails before launch. */
function looksLikeMusicXml(xml: string): boolean {
  const withoutDeclaration = xml.slice(xml.indexOf("<"));
  return /<(score-partwise|score-timewise)[\s>]/.test(withoutDeclaration);
}

export async function renderSheet(opts: RenderSheetOptions): Promise<void> {
  const xml = await readFile(opts.inputPath, "utf8").catch(
    (err: NodeJS.ErrnoException) => {
      throw new RenderError(
        `cannot read input ${opts.inputPath}: ${err.code ?? err.message}`,
      );
    },
  );
  if (!looksLikeMusicXml(xml)) {
    throw new RenderError(
      `input is not MusicXML (expected a score-partwise or score-timewise root): ${opts.inputPath}`,
    );
  }

  let browser;
  try {
    browser = await chromium.launch({ headless: true });
  } catch (err) {
    throw new RenderError(
      `cannot launch Chromium (install the Playwright browser with ` +
        `\`npx playwright install chromium\`): ${String(err)}`,
    );
  }

  try {
    const page = await browser.newPage();
    const pageErrors: string[] = [];
    page.on("pageerror", (err) => pageErrors.push(String(err)));

    await page.goto(VIEWER_URL, { waitUntil: "load" });

    if (opts.format === "svg") {
      const svg = await page.evaluate(
        (musicXml) => window.renderMusicXml(musicXml),
        xml,
      );
      if (typeof svg !== "string" || !svg.includes("<svg")) {
        throw new RenderError(
          `OSMD produced no SVG markup` +
            (pageErrors.length > 0 ? `: ${pageErrors.join("; ")}` : ""),
        );
      }
      await writeFile(opts.outputPath, svg, "utf8");
    } else {
      await page.evaluate((musicXml) => window.renderMusicXml(musicXml), xml);
      const container = page.locator("#osmd-container");
      await container.screenshot({ path: opts.outputPath });
    }
  } catch (err) {
    if (err instanceof RenderError) throw err;
    throw new RenderError(`render failed: ${String(err)}`);
  } finally {
    await browser.close();
  }
}