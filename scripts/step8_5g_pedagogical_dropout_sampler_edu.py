#!/usr/bin/env python3
"""
Step 8.5g — Qualitative sampler for pedagogical dropout cases.

Purpose
-------
Turn the loss-audit outputs into a manageable qualitative reading pack. This is
for close reading, not estimation. It samples pedagogical cases from key loss
buckets, optionally adds chunk context, and surfaces the sentence/statement
profiles that explain why actors drop out of the final rules-only analysis.

Expected prior step
-------------------
Run step8_5c_pedagogical_actor_loss_audit.py first.

Default inputs
--------------
- data/derived/step8_analysis/pedagogical_actor_loss_audit/
- data/derived/step6_chunks_edu/chunks_edu.jsonl

Default outputs
---------------
- data/derived/step8_analysis/pedagogical_dropout_sampler_edu/
    - pedagogical_dropout_sampler.csv
    - pedagogical_dropout_sampler_profile.csv
    - pedagogical_dropout_sampler_manifest.csv
    - pedagogical_dropout_sampler_summary.md
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import spacy


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def maybe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")



def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL, escapechar="\\")



def textify(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()



def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)



def iter_chunks(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunk_text = (
                obj.get("chunk_text")
                or obj.get("text")
                or obj.get("text_normalized")
                or obj.get("chunk")
                or obj.get("content")
                or ""
            )
            rows.append(
                {
                    "doc_id": obj.get("doc_id") or obj.get("source_doc") or obj.get("document_id") or "",
                    "chunk_id": obj.get("chunk_id") or obj.get("chunk_uid") or obj.get("id") or str(i),
                    "chunk_text": str(chunk_text),
                }
            )
    return pd.DataFrame(rows)



def get_sentencizer():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp



def split_sentences(nlp, text: str) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    return [str(s.text).strip() for s in doc.sents if str(s.text).strip()]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="data/derived/step8_analysis/pedagogical_actor_loss_audit")
    ap.add_argument("--chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/pedagogical_dropout_sampler_edu")
    ap.add_argument("--n-rules-cutoff-per-group", type=int, default=25)
    ap.add_argument("--n-parser-miss-per-group", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    audit_dir = Path(args.audit_dir)

    path_rules = audit_dir / "pedagogical_examples_rules_cutoff.csv"
    path_parser = audit_dir / "pedagogical_examples_sentence_only_or_parser_miss.csv"
    path_retained = audit_dir / "pedagogical_examples_final_retained.csv"

    rules_df = maybe_read_table(path_rules)
    parser_df = maybe_read_table(path_parser)
    retained_df = maybe_read_table(path_retained) if path_retained.exists() else pd.DataFrame()

    chunks = iter_chunks(Path(args.chunks_jsonl))
    chunk_map = {(str(r.doc_id), str(r.chunk_id)): r.chunk_text for r in chunks.itertuples(index=False)}
    nlp = get_sentencizer()

    # Normalized columns expected from step 8.5c outputs
    required = {"pedagogical_group", "doc_id", "chunk_id", "sentence_text"}
    for name, df in [("rules_cutoff", rules_df), ("parser_miss", parser_df)]:
        missing = sorted(list(required - set(df.columns)))
        if missing:
            raise ValueError(f"Missing expected columns in {name}: {missing}")

    manifest_rows = []
    sample_frames = []

    def add_sample(df: pd.DataFrame, source_name: str, n_per_group: int) -> None:
        if df.empty:
            return
        for group, gdf in df.groupby("pedagogical_group"):
            sampled = gdf.sample(n=min(n_per_group, len(gdf)), random_state=args.seed).copy()
            sampled["sample_source"] = source_name
            sample_frames.append(sampled)
            manifest_rows.append(
                {
                    "sample_source": source_name,
                    "pedagogical_group": group,
                    "available_rows": int(len(gdf)),
                    "sampled_rows": int(len(sampled)),
                }
            )

    add_sample(rules_df, "lost_rules_only_cutoff", args.n_rules_cutoff_per_group)
    add_sample(parser_df, "sentence_only_or_parser_miss", args.n_parser_miss_per_group)
    if not retained_df.empty:
        add_sample(retained_df, "final_retained", min(10, args.n_parser_miss_per_group))

    if sample_frames:
        sample_df = pd.concat(sample_frames, ignore_index=True).drop_duplicates()
    else:
        sample_df = pd.DataFrame()

    # Add chunk context when available
    if not sample_df.empty:
        prev_list = []
        next_list = []
        for r in sample_df.itertuples(index=False):
            doc_id = str(r.doc_id)
            chunk_id = str(r.chunk_id)
            sent_idx = int(r.sentence_index_in_chunk) if hasattr(r, "sentence_index_in_chunk") and pd.notna(r.sentence_index_in_chunk) else -1
            chunk_text = chunk_map.get((doc_id, chunk_id), "")
            sents = split_sentences(nlp, chunk_text) if chunk_text else []
            prev_sent = sents[sent_idx - 1] if sent_idx > 0 and sent_idx - 1 < len(sents) else ""
            next_sent = sents[sent_idx + 1] if sent_idx >= 0 and sent_idx + 1 < len(sents) else ""
            prev_list.append(prev_sent)
            next_list.append(next_sent)
        sample_df["prev_sentence"] = prev_list
        sample_df["next_sentence"] = next_list

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["sample_source", "pedagogical_group"])

    profile_cols = [c for c in ["sample_source", "pedagogical_group", "statement_type_candidate", "a_class", "d_class"] if c in sample_df.columns]
    if profile_cols:
        profile_df = sample_df.groupby(profile_cols).size().reset_index(name="count").sort_values("count", ascending=False)
    else:
        profile_df = pd.DataFrame()

    path_sampler = out_dir / "pedagogical_dropout_sampler.csv"
    path_profile = out_dir / "pedagogical_dropout_sampler_profile.csv"
    path_manifest = out_dir / "pedagogical_dropout_sampler_manifest.csv"
    safe_to_csv(sample_df, path_sampler)
    safe_to_csv(profile_df, path_profile)
    safe_to_csv(manifest_df, path_manifest)

    top_profile = profile_df.head(25) if not profile_df.empty else profile_df
    md_lines = [
        "# Pedagogical dropout sampler (education corpus)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a qualitative reading pack sampled from the loss-audit outputs.",
        "",
        "## Sampling manifest",
        build_markdown_table(manifest_df) if not manifest_df.empty else "No rows sampled.",
        "",
        "## Top sample profile cells",
        build_markdown_table(top_profile) if not top_profile.empty else "No profile rows available.",
        "",
        "Use the main sampler CSV for close reading. Start with educators_teachers and students_learners in lost_rules_only_cutoff, then compare against any final_retained cases.",
    ]
    (out_dir / "pedagogical_dropout_sampler_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "audit_dir": str(audit_dir),
            "chunks_jsonl": args.chunks_jsonl,
        },
        "sampling": {
            "n_rules_cutoff_per_group": args.n_rules_cutoff_per_group,
            "n_parser_miss_per_group": args.n_parser_miss_per_group,
            "seed": args.seed,
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        path_sampler,
        path_profile,
        path_manifest,
        out_dir / "pedagogical_dropout_sampler_summary.md",
    ]:
        print(f"  {p}")

    print("\nSampling manifest:")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
