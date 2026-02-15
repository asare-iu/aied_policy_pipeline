#!/usr/bin/env python3
"""
Step 8.3b — Umbrella O (document/section-level sanctions) extraction + linking

Deterministic only.
No A lexical lists/bins.

Inputs
- chunks JSONL with keys including: doc_id, chunk_id, text
- 8.3 statements parquet/csv with: doc_id, chunk_id, sentence_text

Outputs (in --out-dir)
- umbrella_o_blocks.parquet
- statements_with_umbrella_o.parquet
- links_only.parquet  (optional, if --write-links-only)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# Config: cues + patterns
# -----------------------------

SANCTION_HEADING_KWS = [
    "sanction", "sanctions",
    "penalty", "penalties",
    "fine", "fines", "administrative fine", "administrative fines",
    "offence", "offences", "offense", "offenses",
    "enforcement",
    "liability", "liable",
    "revocation", "suspension", "withdrawal",
    "non-compliance", "noncompliance",
    "breach",
]

SANCTION_CUES = [
    "failure to comply", "non-compliance", "noncompliance",
    "liable", "liability", "penalty", "penalties", "fine", "fines",
    "offence", "offences", "offense", "offenses",
    "sanction", "sanctions",
    "revocation", "suspension", "withdrawal",
    "shall be punished", "shall be liable",
    "administrative fine", "administrative fines",
    "subject to a fine", "subject to penalties",
    "may impose", "may be imposed", "imposed by",
    "infringement", "infringements",
]

REF_PATTERNS = [
    r"\b(art\.|article)\s+\d+([a-z])?(\(\d+\))*\b",
    r"\b(sec\.|section)\s+\d+(\.\d+)*(\(\d+\))*\b",
    r"\bchapter\s+([ivxlcdm]+|\d+)\b",
    r"\b(annex|schedule)\s+([ivxlcdm]+|\d+)\b",
]

HEADING_MAX_CHARS = 120

REF_RE = re.compile("|".join(f"({p})" for p in REF_PATTERNS), flags=re.IGNORECASE)


# -----------------------------
# Helpers
# -----------------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def find_refs(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0) for m in REF_RE.finditer(text)]


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if len(line) > HEADING_MAX_CHARS:
        return False
    if line.endswith(":"):
        return True
    if line.isupper() and len(line) >= 4:
        return True
    words = re.findall(r"[A-Za-z]+", line)
    if 1 <= len(words) <= 10:
        caps = sum(1 for w in words if w[0].isupper())
        return caps / max(len(words), 1) >= 0.6
    return False


def heading_has_kw(heading: str) -> bool:
    h = heading.lower()
    return any(kw in h for kw in SANCTION_HEADING_KWS)


def text_has_cue(text: str) -> bool:
    t = text.lower()
    return any(cue in t for cue in SANCTION_CUES)


def cues_in_text(text: str) -> List[str]:
    t = text.lower()
    return [cue for cue in SANCTION_CUES if cue in t]


def guess_o_type(text: str) -> str:
    t = text.lower()
    if "administrative fine" in t or "fine" in t:
        return "fine"
    if "offence" in t or "offense" in t or "criminal" in t or "punished" in t:
        return "offence"
    if "revocation" in t or "suspension" in t or "withdrawal" in t:
        return "revocation_or_suspension"
    if "liab" in t:
        return "liability"
    if "sanction" in t or "penalt" in t or "infringement" in t:
        return "sanction_or_penalty"
    return "enforcement_or_other"


@dataclass
class UmbrellaBlock:
    doc_id: str
    chunk_id: str
    block_id: str
    heading_text: str
    block_text: str
    o_type: str
    cues: List[str]
    refs: List[str]
    method: str
    confidence: str

# -----------------------------
# Streaming JSONL
# -----------------------------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------------
# Extraction: umbrella blocks
# -----------------------------

def extract_blocks_from_text(doc_id: str, chunk_id: str, text: str) -> List[UmbrellaBlock]:
    blocks: List[UmbrellaBlock] = []
    if not text:
        return blocks

    lines = [ln.rstrip() for ln in text.splitlines()]

    # 1) Heading-based blocks
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if looks_like_heading(ln) and heading_has_kw(ln):
            heading = ln.rstrip(":").strip()
            start = i + 1
            j = start
            while j < len(lines):
                if looks_like_heading(lines[j]) and j != i:
                    break
                j += 1

            body = normalize_ws("\n".join(lines[start:j]).strip())
            if body and (text_has_cue(body) or heading_has_kw(heading)):
                b_id = f"{doc_id}::{chunk_id}::H{i}"
                blocks.append(
                    UmbrellaBlock(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        block_id=b_id,
                        heading_text=heading,
                        block_text=body,
                        o_type=guess_o_type(body),
                        cues=cues_in_text(body),
                        refs=find_refs(body),
                        method="heading_block",
                        confidence="high",
                    )
                )
            i = j
        else:
            i += 1

    # 2) Fallback: cue paragraphs if no heading blocks found
    if not blocks:
        paras = [normalize_ws(p) for p in re.split(r"\n\s*\n", text) if normalize_ws(p)]
        for idx, p in enumerate(paras):
            if text_has_cue(p):
                b_id = f"{doc_id}::{chunk_id}::P{idx}"
                blocks.append(
                    UmbrellaBlock(
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        block_id=b_id,
                        heading_text="",
                        block_text=p,
                        o_type=guess_o_type(p),
                        cues=cues_in_text(p),
                        refs=find_refs(p),
                        method="cue_paragraph",
                        confidence="medium",
                    )
                )

    return blocks


def extract_umbrella_blocks_streaming(
    chunks_jsonl: Path,
    report_every: int = 50_000,
    max_chunks: Optional[int] = None,
) -> pd.DataFrame:
    out_rows: List[dict] = []
    t0 = time.time()

    for n, ch in enumerate(iter_jsonl(chunks_jsonl), start=1):
        doc_id = str(ch.get("doc_id", ""))
        chunk_id = str(ch.get("chunk_id", ""))
        text = ch.get("text") or ""

        blocks = extract_blocks_from_text(doc_id, chunk_id, text)
        for b in blocks:
            out_rows.append(
                {
                    "doc_id": b.doc_id,
                    "chunk_id": b.chunk_id,
                    "block_id": b.block_id,
                    "heading_text": b.heading_text,
                    "block_text": b.block_text,
                    "o_umbrella_type": b.o_type,
                    "o_umbrella_cues": "|".join(b.cues) if b.cues else "",
                    "o_umbrella_refs": "|".join(b.refs) if b.refs else "",
                    "o_umbrella_method": b.method,
                    "o_umbrella_block_confidence": b.confidence,
                }
            )

        if report_every and n % report_every == 0:
            dt = time.time() - t0
            print(f"[extract] chunks={n:,} blocks={len(out_rows):,} elapsed={dt/60:.1f}m")

        if max_chunks and n >= max_chunks:
            break

    dt = time.time() - t0
    print(f"[extract] DONE chunks={n:,} blocks={len(out_rows):,} elapsed={dt/60:.1f}m")
    return pd.DataFrame(out_rows)


# -----------------------------
# Linking: statements -> blocks
# -----------------------------

def build_blocks_index(umb: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Returns: doc_id -> list of block dicts with:
      block_id, chunk_id, method_rank, refs_set, ... (fields)
    """
    by_doc: Dict[str, List[dict]] = {}
    if umb.empty:
        return by_doc

    # precompute refs sets
    refs_series = umb["o_umbrella_refs"].fillna("").astype(str)
    refs_sets: List[Set[str]] = []
    for s in refs_series.tolist():
        if not s:
            refs_sets.append(set())
        else:
            refs_sets.append(set([r for r in s.split("|") if r]))
    umb = umb.copy()
    umb["_refs_set"] = refs_sets
    umb["_method_rank"] = umb["o_umbrella_method"].apply(lambda m: 2 if m == "heading_block" else 1)

    for doc_id, g in umb.groupby("doc_id"):
        recs = g.to_dict(orient="records")
        by_doc[str(doc_id)] = recs
    return by_doc


def choose_best_block_for_statement(
    doc_blocks: List[dict],
    stmt_doc_id: str,
    stmt_chunk_id: str,
    stmt_refs: Set[str],
) -> Tuple[Optional[str], str, str]:
    """
    Returns: (block_id, confidence, link_method)
    confidence: high|medium|low|none
    """
    if not doc_blocks:
        return None, "none", ""

    # HIGH: ref overlap
    if stmt_refs:
        best = None
        best_overlap = 0
        best_rank = -1
        for b in doc_blocks:
            bref: Set[str] = b.get("_refs_set", set())
            overlap = len(stmt_refs & bref) if bref else 0
            if overlap > 0:
                rank = int(b.get("_method_rank", 1))
                if overlap > best_overlap or (overlap == best_overlap and rank > best_rank):
                    best = b
                    best_overlap = overlap
                    best_rank = rank
        if best is not None:
            return str(best["block_id"]), "high", "ref_overlap"

    # MEDIUM: same chunk_id (structural proximity)
    same_chunk = [b for b in doc_blocks if str(b.get("chunk_id", "")) == str(stmt_chunk_id)]
    if same_chunk:
        same_chunk.sort(key=lambda b: int(b.get("_method_rank", 1)), reverse=True)
        return str(same_chunk[0]["block_id"]), "medium", "same_chunk"

    # LOW: doc fallback (strongest method)
    doc_blocks.sort(key=lambda b: int(b.get("_method_rank", 1)), reverse=True)
    return str(doc_blocks[0]["block_id"]), "low", "doc_fallback"


def link_statements_to_blocks(stmts: pd.DataFrame, umb: pd.DataFrame, report_every: int = 50_000) -> pd.DataFrame:
    t0 = time.time()
    blocks_index = build_blocks_index(umb)

    out = stmts.copy()
    out["stmt_refs"] = out["sentence_text"].astype(str).apply(lambda s: find_refs(s))
    out["stmt_refs_set"] = out["stmt_refs"].apply(lambda xs: set(xs) if xs else set())

    block_ids: List[Optional[str]] = []
    confs: List[str] = []
    methods: List[str] = []

    for i, row in enumerate(out.itertuples(index=False), start=1):
        doc_id = str(getattr(row, "doc_id"))
        chunk_id = str(getattr(row, "chunk_id"))
        refs_set = getattr(row, "stmt_refs_set")

        doc_blocks = blocks_index.get(doc_id, [])
        b_id, conf, meth = choose_best_block_for_statement(doc_blocks, doc_id, chunk_id, refs_set)

        block_ids.append(b_id)
        confs.append(conf)
        methods.append(meth)

        if report_every and i % report_every == 0:
            dt = time.time() - t0
            print(f"[link] statements={i:,}/{len(out):,} elapsed={dt/60:.1f}m")

    out["o_umbrella_block_id"] = block_ids
    out["o_umbrella_confidence"] = confs
    out["o_umbrella_link_method"] = methods
    out["o_umbrella_present"] = out["o_umbrella_block_id"].notna()

    # audit-friendly join of refs
    out["stmt_refs"] = out["stmt_refs"].apply(lambda xs: "|".join(xs) if xs else "")
    out = out.drop(columns=["stmt_refs_set"])

    dt = time.time() - t0
    print(f"[link] DONE statements={len(out):,} elapsed={dt/60:.1f}m")
    return out


def merge_block_fields(linked: pd.DataFrame, umb: pd.DataFrame) -> pd.DataFrame:
    if umb.empty:
        return linked

    umb_small = umb.rename(
        columns={
            "block_id": "o_umbrella_block_id",
            "heading_text": "o_umbrella_heading",
            "block_text": "o_umbrella_text",
        }
    )[
        [
            "o_umbrella_block_id",
            "o_umbrella_heading",
            "o_umbrella_text",
            "o_umbrella_type",
            "o_umbrella_cues",
            "o_umbrella_refs",
            "o_umbrella_method",
            "o_umbrella_block_confidence",
        ]
    ]

    return linked.merge(umb_small, on="o_umbrella_block_id", how="left")


# -----------------------------
# IO
# -----------------------------

def read_table(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, escapechar="\\")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-jsonl", required=True)
    ap.add_argument("--statements", required=True)
    ap.add_argument("--statements-format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-format", default="parquet", choices=["parquet", "csv"])
    ap.add_argument("--report-every", type=int, default=50_000)
    ap.add_argument("--max-chunks", type=int, default=0, help="Debug: stop after N chunks (0=all)")
    ap.add_argument("--write-merged", action="store_true", help="Write merged statements+block text")
    ap.add_argument("--write-links-only", action="store_true", help="Write only linking columns (no merge)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_jsonl = Path(args.chunks_jsonl)
    statements_path = Path(args.statements)

    print("[main] extracting umbrella O blocks…")
    umb = extract_umbrella_blocks_streaming(
        chunks_jsonl,
        report_every=args.report_every,
        max_chunks=(args.max_chunks or None),
    )

    umb_out = out_dir / f"umbrella_o_blocks.{args.out_format}"
    write_table(umb, umb_out, args.out_format)
    print(f"[main] saved → {umb_out} (rows={len(umb):,})")

    print("[main] linking statements → umbrella blocks…")
    stmts = read_table(statements_path, args.statements_format)
    linked = link_statements_to_blocks(stmts, umb, report_every=args.report_every)

    # links-only output (small)
    if args.write_links_only:
        cols = [
            "doc_id", "chunk_id", "sentence_index_in_chunk", "sentence_text",
            "o_umbrella_present", "o_umbrella_block_id", "o_umbrella_confidence", "o_umbrella_link_method", "stmt_refs"
        ]
        links_only = linked[[c for c in cols if c in linked.columns]].copy()
        links_out = out_dir / f"links_only.{args.out_format}"
        write_table(links_only, links_out, args.out_format)
        print(f"[main] saved → {links_out} (rows={len(links_only):,})")

    # merged output (with block text/type)
    if args.write_merged:
        merged = merge_block_fields(linked, umb)
        merged_out = out_dir / f"statements_with_umbrella_o.{args.out_format}"
        write_table(merged, merged_out, args.out_format)
        print(f"[main] saved → {merged_out} (rows={len(merged):,})")
    else:
        linked_out = out_dir / f"statements_linked_umbrella_o.{args.out_format}"
        write_table(linked, linked_out, args.out_format)
        print(f"[main] saved → {linked_out} (rows={len(linked):,})")

    # quick audit
    pct = round(100 * linked["o_umbrella_present"].mean(), 2) if len(linked) else 0.0
    conf = linked["o_umbrella_confidence"].value_counts(dropna=False).to_dict() if len(linked) else {}
    print(f"[audit] umbrella linked % = {pct}")
    print(f"[audit] confidence counts = {conf}")


if __name__ == "__main__":
    main()
