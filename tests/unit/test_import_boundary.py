"""The note-writing packages cannot reach a model.

`docs/model-fine-tuning.md` argues that no LLM is reachable from the code
that writes, voices, lints or renders a note, and invites the reader to
check it with a grep:

    grep -rn "saimc.llm" src/saimc/compose src/saimc/render src/saimc/release

A grep is not a guard. It misses a relative import written as
`from ..llm import parse`, and nothing runs it — so the claim holds only
until someone edits the code and forgets the document. This module is that
claim as a test.

It matters now because the interactive session work introduces a conductor
that has to stay on the *outside* of these packages: the engine is the only
writer of notes, and a path from it back to a model would make the
determinism contract conditional on the model's behaviour.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "saimc"

GUARDED_PACKAGES = ("compose", "render", "release")

FORBIDDEN_ROOTS = ("saimc.llm", "saimc.session")
"""Modules the guarded packages must not reach.

`saimc.session` does not exist yet. It is listed ahead of the code on
purpose: the guard is worth having before the import it forbids is
available to write.
"""


def _guarded_modules() -> list[Path]:
    modules: list[Path] = []
    for package in GUARDED_PACKAGES:
        modules.extend(sorted((SRC / package).rglob("*.py")))
    return modules


def _module_name(path: Path) -> str:
    """`src/saimc/compose/engine.py` -> `saimc.compose.engine`."""
    relative = path.relative_to(REPO_ROOT / "src")
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module_name: str, path: Path) -> str:
    """The package a module's relative imports are resolved against."""
    if path.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _resolve(node: ast.ImportFrom, package: str) -> str:
    """The absolute module an `ImportFrom` names, relative forms included.

    `from ..llm import parse` inside `saimc.compose.engine` is `saimc.llm`
    — exactly the form a grep for the literal text does not catch.
    """
    if node.level == 0:
        return node.module or ""
    parts = package.split(".")
    kept = parts[: len(parts) - (node.level - 1)]
    if node.module:
        kept.extend(node.module.split("."))
    return ".".join(kept)


def _imported_modules(tree: ast.AST, package: str) -> list[str]:
    """Every module named by an import, static or dynamically.

    The dynamic forms are included because `import_module("saimc.llm")`
    reaches the same module as `import saimc.llm` while being invisible to
    both a grep and a check that only walks `ast.Import`. Only literal
    string arguments are read, so a docstring mentioning a module name is
    not mistaken for an import of it.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(_resolve(node, package))
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            # `importlib.import_module(x)`, a bare `import_module(x)` after
            # `from importlib import import_module`, and `__import__(x)` all
            # reach the same module; only the first was caught before.
            is_dynamic_import = (
                isinstance(func, ast.Name) and func.id in {"__import__", "import_module"}
            ) or (isinstance(func, ast.Attribute) and func.attr == "import_module")
            argument = node.args[0]
            if (
                is_dynamic_import
                and isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ):
                found.append(argument.value)
    return found


def _is_forbidden(module: str) -> bool:
    return any(module == root or module.startswith(f"{root}.") for root in FORBIDDEN_ROOTS)


@pytest.mark.parametrize("path", _guarded_modules(), ids=lambda path: str(path.relative_to(SRC)))
def test_no_note_writing_module_reaches_a_model(path: Path) -> None:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = sorted(
        {
            module
            for module in _imported_modules(tree, _package_of(module_name, path))
            if _is_forbidden(module)
        }
    )
    assert not offenders, (
        f"{module_name} imports {', '.join(offenders)}, which puts a model behind the "
        "code that writes notes. A model may propose a spec or a plan; it must not be "
        "reachable from the engine, the linter or the renderers. See "
        "docs/model-fine-tuning.md."
    )


class TestGuardIsActive:
    """A guard that cannot fail guards nothing."""

    def test_it_catches_the_forms_a_grep_misses(self) -> None:
        source = (
            "from ..llm import parse\n"
            "import saimc.session.conductor\n"
            "from importlib import import_module\n"
            "import_module('saimc.llm.base')\n"
        )
        offenders = sorted(
            {
                module
                for module in _imported_modules(ast.parse(source), "saimc.compose")
                if _is_forbidden(module)
            }
        )
        assert offenders == [
            "saimc.llm",
            "saimc.llm.base",
            "saimc.session.conductor",
        ]

    def test_it_passes_imports_that_are_allowed(self) -> None:
        """The same walk over an ordinary module finds nothing to report.

        Without this, a checker that returned everything — or a
        `FORBIDDEN_ROOTS` typo — would pass the test above by accident.
        """
        source = (
            "import saimc.compose.score\n"
            "from .. import canonical\n"
            "from .forms import ChordSlot\n"
            "from saimc.render import audio\n"
        )
        offenders = [
            module
            for module in _imported_modules(ast.parse(source), "saimc.compose")
            if _is_forbidden(module)
        ]
        assert not offenders
