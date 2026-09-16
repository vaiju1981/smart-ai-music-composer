#!/usr/bin/env python3
"""List the presets inside a SoundFont 2 file.

Dedicated (non-GM) soundfonts number their presets however their author
liked, so wiring an instrument to a dedicated font means knowing which
bank:preset pair plays it. This script reads the `phdr` chunk directly
(no soundfont library needed) and prints every preset as
`bank:program  name`, sorted the way FluidSynth indexes them.

Usage:
    python scripts/sf2_presets.py assets/soundfonts/mfa_boston_1.sf2
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def _chunks(data: bytes) -> list[tuple[str, bytes]]:
    """Yield the top-level RIFF chunks of an SF2 file as (id, payload)."""
    if data[:4] != b"RIFF" or data[8:12] != b"sfbk":
        raise SystemExit(f"{data[:12]!r} is not a SoundFont 2 (RIFF/sfbk) file")
    pos = 12
    chunks: list[tuple[str, bytes]] = []
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        chunks.append((cid, data[pos + 8 : pos + 8 + size]))
        pos += 8 + size + (size % 2)  # chunks are word-aligned
    return chunks


def _list_subchunks(payload: bytes) -> list[tuple[str, bytes]]:
    pos = 4  # skip the LIST type id
    subchunks: list[tuple[str, bytes]] = []
    while pos + 8 <= len(payload):
        cid = payload[pos : pos + 4].decode("ascii", "replace")
        size = struct.unpack_from("<I", payload, pos + 4)[0]
        subchunks.append((cid, payload[pos + 8 : pos + 8 + size]))
        pos += 8 + size + (size % 2)
    return subchunks


def presets(path: Path) -> list[tuple[str, int, int]]:
    """Return (name, bank, program) for every preset in the font."""
    for cid, payload in _chunks(path.read_bytes()):
        if cid != "LIST":
            continue
        list_type = payload[:4].decode("ascii", "replace")
        if list_type != "pdta":
            continue
        for sub, sub_payload in _list_subchunks(payload):
            if sub != "phdr":
                continue
            found: list[tuple[str, int, int]] = []
            record_size = 38  # 20-byte name + 4 u16 + 3 u32
            for off in range(0, len(sub_payload) - record_size + 1, record_size):
                raw_name = sub_payload[off : off + 20]
                name = raw_name.split(b"\x00", 1)[0].decode("ascii", "replace")
                preset, bank = struct.unpack_from("<HH", sub_payload, off + 20)
                found.append((name, bank, preset))
            # The last record is the end sentinel ("EOD"/"EOP"); it
            # carries no playable preset.
            return [p for p in found if p[0].upper() not in ("EOD", "EOP")]
    raise SystemExit("no phdr chunk found — file is not a complete SF2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sf2", type=Path, help="path to a .sf2 file")
    args = parser.parse_args()
    rows = presets(args.sf2)
    print(f"{args.sf2}: {len(rows)} presets")
    for name, bank, program in sorted(rows, key=lambda p: (p[1], p[2])):
        print(f"  {bank:>3}:{program:<4} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
