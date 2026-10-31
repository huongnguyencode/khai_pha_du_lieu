"""Small, dependency-free reader for the numeric/character SAS XPORT v5 files.

The project environment currently has an incompatible pandas/numpy pair.  NHANES
XPT files use the simple v5 transport layout, so keeping this reader local also
makes the raw-to-canonical conversion reproducible without a binary dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


NAMESTR_MARKER = b"HEADER RECORD*******NAMESTR HEADER RECORD!!!!!!!"


@dataclass(frozen=True)
class Variable:
    name: str
    kind: int
    length: int
    offset: int
    label: str


def _ibm_float(raw: bytes) -> float:
    if raw == b"\x00" * 8:
        return 0.0
    # SAS numeric missing values are '.', '_' or '.A' style codes followed by 0s.
    if raw[1:] == b"\x00" * 7 and raw[0] in b"._ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return math.nan
    sign = -1.0 if raw[0] & 0x80 else 1.0
    exponent = (raw[0] & 0x7F) - 64
    fraction = int.from_bytes(raw[1:], "big") / (1 << 56)
    return sign * (16.0**exponent) * fraction


def read_xport(path: str | Path) -> tuple[list[Variable], list[dict[str, object]]]:
    """Read one NHANES XPT file and return its dictionary and records."""
    payload = Path(path).read_bytes()
    header_pos = payload.index(NAMESTR_MARKER)
    variable_count = int(payload[header_pos + 48 : header_pos + 58])
    names_pos = header_pos + 80
    variables: list[Variable] = []
    for index in range(variable_count):
        block = payload[names_pos + index * 140 : names_pos + (index + 1) * 140]
        variables.append(
            Variable(
                name=block[8:16].decode("ascii").rstrip(),
                kind=int.from_bytes(block[0:2], "big"),
                length=int.from_bytes(block[4:6], "big"),
                offset=int.from_bytes(block[84:88], "big"),
                label=block[16:56].decode("latin1").rstrip(),
            )
        )

    observations_pos = ((names_pos + variable_count * 140 + 79) // 80) * 80
    if b"OBS" not in payload[observations_pos : observations_pos + 80]:
        raise ValueError(f"OBS header not found in {path}")
    observations_pos += 80
    row_length = sum(variable.length for variable in variables)
    rows: list[dict[str, object]] = []
    for pos in range(observations_pos, len(payload) - row_length + 1, row_length):
        block = payload[pos : pos + row_length]
        # XPORT pads the last physical 80-byte record with spaces.
        if all(byte in (0, 32) for byte in block):
            break
        row: dict[str, object] = {}
        for variable in variables:
            raw = block[variable.offset : variable.offset + variable.length]
            if variable.kind == 1:
                value: object = _ibm_float(raw)
            else:
                value = raw.decode("latin1").rstrip()
            row[variable.name] = value
        seqn = row.get("SEQN")
        if not isinstance(seqn, float) or not math.isfinite(seqn) or seqn < 100000:
            break
        rows.append(row)
    return variables, rows

