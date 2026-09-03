"""Allow `python -m saimc ...` invocations."""

from __future__ import annotations

import sys


def main() -> int:
    print("saimc CLI entrypoint — not yet implemented.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
