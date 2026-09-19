#!/usr/bin/env python3
"""Check that every `[project.scripts]` entry is installed and runnable.

A `[project.scripts]` target can be wrong and nothing says so: pip writes a shim
for whatever the table names, and the shim only fails when someone actually runs
it. Two defects of that shape have shipped here — an entry naming a module that
moved, and, the one that bit, the four typer CLIs naming their
`@app.command()`-decorated `main` instead of the `_entry` that calls `app()`.
Calling a decorated command directly bypasses typer's parsing, so every
invocation died on the raw `typer.Option(...)` defaults while every import
succeeded and every test passed.

So four things are checked per entry: the shim exists, its module imports, the
named attribute is there, and — for the typer CLIs — `--help` exits 0. That last
one is the only check that *executes* the entry point, and the only one that
catches the defect above.

It is applied to the typer CLIs alone, and deliberately: `--help` is safe there
because click exits before the body runs, whereas a plain `main()` may perform
its whole job when invoked — `saimc-build-ffmpeg` would start building ffmpeg —
so those are checked by resolution only. Which scripts get which treatment is
read off the module's own `app` attribute rather than from a list of names, so a
new typer CLI is covered the day it is added.

Called by `run.sh bootstrap` and by CI. The shims are looked for beside the
interpreter running this script — `.venv/bin` under `run.sh`, and the CI
interpreter's own bin directory in CI — so the same code checks both without
either knowing about the other. Pass a directory to check a venv you did not
create. Exits 1 and names each problem if anything fails.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_scripts() -> dict[str, str]:
    """The console-script table, as `name -> "module:attr"`."""
    document = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts: dict[str, str] = document["project"]["scripts"]
    return scripts


def _is_typer_cli(module: object) -> bool:
    """Whether this module is a typer app, i.e. whether `--help` is safe to run."""
    app = getattr(module, "app", None)
    return type(app).__module__.split(".")[0] == "typer"


def check(scripts: dict[str, str], bin_dir: Path) -> list[str]:
    """Every problem found, each naming the script it is about."""
    problems: list[str] = []
    for name, target in scripts.items():
        module_name, _, attr = target.partition(":")
        executable = bin_dir / name
        if not executable.exists():
            problems.append(f"{name}: no {executable} (entry points are stale — reinstall)")
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(f"{name}: cannot import {module_name}: {exc}")
            continue
        if not hasattr(module, attr):
            problems.append(f"{name}: {module_name} has no {attr!r}")
            continue
        if _is_typer_cli(module):
            result = subprocess.run([str(executable), "--help"], capture_output=True, text=True)
            if result.returncode != 0:
                last = (result.stderr or result.stdout).strip().splitlines()
                problems.append(f"{name}: --help failed: {last[-1] if last else result.returncode}")
    return problems


def main(argv: list[str]) -> int:
    bin_dir = Path(argv[1]) if len(argv) > 1 else Path(sys.executable).parent
    scripts = load_scripts()
    problems = check(scripts, bin_dir)
    print(f"scripts: {len(scripts) - len(problems)}/{len(scripts)} verified")
    for problem in problems:
        print(f"  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
