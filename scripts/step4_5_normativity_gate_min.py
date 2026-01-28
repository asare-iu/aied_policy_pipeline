#!/usr/bin/env python3
"""
Step 4.5 (minimal): Normativity / institutional-statement gate.

Input:
- data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl

Outputs:
- data/derived/step4_5_normativity_gate/chunks_normative_primary.jsonl
- data/derived/step4_5_normativity_gate/gate_summary.json

Gate logic (primary candidates):
- Keep if any of these are true:
  1) has_deontic
  2) has_authority_delegation
  3) has_enforcement
  4) has_scope_applicability
  5) has_definition
  6) has_conditional AND (has_deontic OR has_authority_delegation)

Rationale:
- Deontics, authority/delegation, enforcement, scope, and definitions are strong institutional markers.
- Conditionals alone are too broad; combined with deontic/authority they signal structured rules.
- Info/reporting and monitoring/audit are intentionally NOT sufficient on their own to qualify.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from tqdm import tqdm


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_primary(tags: Dict[str, Any]) -> bool:
    d = bool(tags.get("has_deontic"))
    a = bool(tags.get("has_authority_delegation"))
    e = bool(tags.get("has_enforcement"))
    s = bool(tags.get("has_scope_applicability"))
    defin = bool(tags.get("has_definition"))
    c = bool(tags.get("has_conditional"))

    return d or a or e or s or defin or (c and (d or a))


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 4.5: normativity gate (primary candidates).")
    parser.add_argument(
        "--input",
        type=str,
        default="data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl",
        help="Tagged chunks JSONL from Step 4.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/step4_5_normativity_gate",
        help="Directory for gate outputs.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_primary = out_dir / "chunks_normative_primary.jsonl"
    out_summary = out_dir / "gate_summary.json"

    total = 0
    kept = 0

    # Stream read/write so this stays memory-light.
    with out_primary.open("w", encoding="utf-8") as out:
        for r in tqdm(read_jsonl(in_path), desc="Step 4.5: gating", unit="chunk"):
            total += 1
            tags = r.get("tags", {})
            if is_primary(tags):
                kept += 1
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_chunks": total,
        "kept_primary": kept,
        "kept_primary_pct": kept / total if total else 0.0,
        "gate_version": "primary_v1",
        "gate_notes": "Info/reporting and monitoring/audit alone are not sufficient markers.",
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Step 4.5 complete | total={total} kept_primary={kept} ({kept/total:.2%})")
    print(f"Wrote: {out_primary}")
    print(f"Wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
