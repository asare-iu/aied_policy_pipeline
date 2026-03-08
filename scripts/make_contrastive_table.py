#!/usr/bin/env python3
from __future__ import annotations

"""
Build one clean contrastive table across three corpora:
  1) Full corpus
  2) Education-relevant subset
  3) Education-in-title subset

Primary use case:
- Run inside the full repo after Step 8 outputs exist.
- Produces a single wide CSV that is easy to drop into the dissertation,
  plus a long-form CSV and markdown version.

Default inputs are aligned to the repository structure used in the project.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import pandas as pd


@dataclass
class CorpusSpec:
    label: str
    chunks_path: Path
    igt_path: Path
    doc_ids_path: Optional[Path] = None


def pct(num: float, den: float) -> float:
    return round((100.0 * num / den), 2) if den else 0.0


def rate(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return 0.0
    return round(100.0 * float(series.fillna("<NA>").astype(str).eq(value).mean()), 2)


def truth_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    s = series.fillna(False)
    if s.dtype == bool:
        return round(100.0 * float(s.mean()), 2)
    return round(100.0 * float(s.astype(str).str.lower().isin(["true", "1", "yes"]).mean()), 2)


def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_doc_id(obj: dict) -> Optional[str]:
    for key in ("doc_id", "document_id", "docid"):
        if key in obj and obj[key] is not None:
            return str(obj[key])
    return None


def count_jsonl_and_doc_ids(path: Path) -> tuple[int, Set[str]]:
    n = 0
    doc_ids: Set[str] = set()
    for obj in iter_jsonl(path):
        n += 1
        did = extract_doc_id(obj)
        if did:
            doc_ids.add(did)
    return n, doc_ids


def load_doc_ids_txt(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_inventory(path: Path) -> pd.DataFrame:
    inv = pd.read_csv(path)
    cols = {c.lower(): c for c in inv.columns}

    doc_col = None
    for c in ("doc_id", "docid", "document_id"):
        if c in cols:
            doc_col = cols[c]
            break
    if doc_col is None:
        raise ValueError(f"No doc_id column found in inventory: {path}")

    country_col = None
    for c in ("country", "jurisdiction", "country_name"):
        if c in cols:
            country_col = cols[c]
            break

    title_col = None
    for c in ("title", "document_title", "doc_title", "name"):
        if c in cols:
            title_col = cols[c]
            break

    keep = [doc_col]
    if country_col:
        keep.append(country_col)
    if title_col:
        keep.append(title_col)

    out = inv[keep].copy().rename(columns={doc_col: "doc_id"})
    if country_col:
        out = out.rename(columns={country_col: "country"})
    if title_col:
        out = out.rename(columns={title_col: "title"})
    out["doc_id"] = out["doc_id"].astype(str)
    if "country" in out.columns:
        out["country"] = out["country"].fillna("").astype(str)
    return out.drop_duplicates(subset=["doc_id"])


def summarize_igt(df: pd.DataFrame) -> Dict[str, float | int]:
    n = len(df)
    out: Dict[str, float | int] = {
        "statements_total": int(n),
        "stmt_unique_docs": int(df["doc_id"].nunique()) if "doc_id" in df.columns else 0,
    }

    if "a_class" in df.columns:
        out["explicit_A_pct"] = rate(df["a_class"], "explicit")
        out["implicit_A_pct"] = rate(df["a_class"], "implicit")
        out["any_entity_A_pct"] = rate(df["a_class"], "any_entity")
    else:
        out["explicit_A_pct"] = 0.0
        out["implicit_A_pct"] = 0.0
        out["any_entity_A_pct"] = 0.0

    if "d_lemma" in df.columns:
        out["D_present_pct"] = round(100.0 * float(df["d_lemma"].notna().mean()), 2) if n else 0.0
    elif "d_class" in df.columns:
        out["D_present_pct"] = round(100.0 * float(df["d_class"].notna().mean()), 2) if n else 0.0
    else:
        out["D_present_pct"] = 0.0

    if "d_class" in df.columns:
        out["obligation_pct"] = rate(df["d_class"], "obligation")
        out["permission_pct"] = rate(df["d_class"], "permission")
        out["prohibition_pct"] = rate(df["d_class"], "prohibition")
    else:
        out["obligation_pct"] = 0.0
        out["permission_pct"] = 0.0
        out["prohibition_pct"] = 0.0

    if "statement_type_candidate" in df.columns:
        out["rule_candidate_pct"] = rate(df["statement_type_candidate"], "rule_candidate")
        out["norm_candidate_pct"] = rate(df["statement_type_candidate"], "norm_candidate")
        out["strategy_candidate_pct"] = rate(df["statement_type_candidate"], "strategy_candidate")
        out["other_low_confidence_pct"] = rate(df["statement_type_candidate"], "other_low_confidence")
    else:
        out["rule_candidate_pct"] = 0.0
        out["norm_candidate_pct"] = 0.0
        out["strategy_candidate_pct"] = 0.0
        out["other_low_confidence_pct"] = 0.0

    if "o_local_present" in df.columns:
        out["o_scope_present_pct"] = truth_rate(df["o_local_present"])
    else:
        out["o_scope_present_pct"] = 0.0

    return out


def summarize_corpus(spec: CorpusSpec, inventory: Optional[pd.DataFrame]) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {"corpus": spec.label}

    chunk_doc_ids: Set[str] = set()
    if spec.chunks_path.exists():
        n_chunks, chunk_doc_ids = count_jsonl_and_doc_ids(spec.chunks_path)
        row["chunks_total"] = int(n_chunks)
    else:
        row["chunks_total"] = 0

    if spec.doc_ids_path and spec.doc_ids_path.exists():
        doc_ids = load_doc_ids_txt(spec.doc_ids_path)
    elif chunk_doc_ids:
        doc_ids = chunk_doc_ids
    else:
        doc_ids = set()

    if spec.igt_path.exists():
        igt = maybe_read_table(spec.igt_path)
        row.update(summarize_igt(igt))
        if not doc_ids and "doc_id" in igt.columns:
            doc_ids = set(igt["doc_id"].dropna().astype(str).unique().tolist())
    else:
        row.update(
            {
                "statements_total": 0,
                "stmt_unique_docs": 0,
                "explicit_A_pct": 0.0,
                "implicit_A_pct": 0.0,
                "any_entity_A_pct": 0.0,
                "D_present_pct": 0.0,
                "obligation_pct": 0.0,
                "permission_pct": 0.0,
                "prohibition_pct": 0.0,
                "rule_candidate_pct": 0.0,
                "norm_candidate_pct": 0.0,
                "strategy_candidate_pct": 0.0,
                "other_low_confidence_pct": 0.0,
                "o_scope_present_pct": 0.0,
            }
        )

    row["unique_docs"] = int(len(doc_ids))

    if inventory is not None and len(doc_ids) > 0 and "country" in inventory.columns:
        row["unique_countries"] = int(inventory[inventory["doc_id"].isin(doc_ids)]["country"].replace("", pd.NA).dropna().nunique())
    else:
        row["unique_countries"] = 0

    row["statements_per_100_chunks"] = pct(float(row["statements_total"]), float(row["chunks_total"]))
    return row


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(id_vars=["corpus"], var_name="metric", value_name="value")
    return long_df.sort_values(["metric", "corpus"]).reset_index(drop=True)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    pretty = df.copy()
    for col in pretty.columns:
        if col == "corpus":
            continue
        # render floats nicely but leave ints alone
        if pd.api.types.is_float_dtype(pretty[col]):
            pretty[col] = pretty[col].map(lambda x: f"{x:.2f}")
    try:
        path.write_text(pretty.to_markdown(index=False) + "\n", encoding="utf-8")
    except Exception:
        path.write_text(pretty.to_csv(index=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="data/derived/step0_document_inventory_deduped.csv")
    ap.add_argument("--full-chunks", default="data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl")
    ap.add_argument("--edu-chunks", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--title-chunks", default="data/derived/step7_chunks_title_edu/chunks_title_edu_allchunks.jsonl")
    ap.add_argument("--full-igt", default="data/derived/step8_igt_full/igt_statements_full.parquet")
    ap.add_argument("--edu-igt", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--title-igt", default="data/derived/step8_igt_title_edu/igt_statements_full.parquet")
    ap.add_argument("--title-doc-ids", default="data/derived/step6b_title_edu/doc_ids_title_tier1plus2.txt")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = Path(args.inventory)
    inventory = load_inventory(inventory_path) if inventory_path.exists() else None

    corpora = [
        CorpusSpec(
            label="Full corpus",
            chunks_path=Path(args.full_chunks),
            igt_path=Path(args.full_igt),
        ),
        CorpusSpec(
            label="Education-relevant",
            chunks_path=Path(args.edu_chunks),
            igt_path=Path(args.edu_igt),
        ),
        CorpusSpec(
            label="Education-in-title",
            chunks_path=Path(args.title_chunks),
            igt_path=Path(args.title_igt),
            doc_ids_path=Path(args.title_doc_ids),
        ),
    ]

    rows = [summarize_corpus(spec, inventory) for spec in corpora]
    wide = pd.DataFrame(rows)

    ordered_cols = [
        "corpus",
        "unique_docs",
        "unique_countries",
        "chunks_total",
        "statements_total",
        "statements_per_100_chunks",
        "explicit_A_pct",
        "implicit_A_pct",
        "any_entity_A_pct",
        "D_present_pct",
        "obligation_pct",
        "permission_pct",
        "prohibition_pct",
        "rule_candidate_pct",
        "norm_candidate_pct",
        "strategy_candidate_pct",
        "other_low_confidence_pct",
        "o_scope_present_pct",
        "stmt_unique_docs",
    ]
    ordered_cols = [c for c in ordered_cols if c in wide.columns]
    wide = wide[ordered_cols]

    long_df = wide_to_long(wide)

    wide_csv = out_dir / "contrastive_table_full_vs_edu_vs_title.csv"
    long_csv = out_dir / "contrastive_table_full_vs_edu_vs_title_long.csv"
    md_path = out_dir / "contrastive_table_full_vs_edu_vs_title.md"

    wide.to_csv(wide_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    write_markdown(wide, md_path)

    print(f"Saved → {wide_csv}")
    print(f"Saved → {long_csv}")
    print(f"Saved → {md_path}")
    print()
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
