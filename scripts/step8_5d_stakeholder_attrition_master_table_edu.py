#!/usr/bin/env python3
"""
Step 8.5d — Master stakeholder attrition table for the education-relevant corpus.

Purpose
-------
Build the clearest comparative baseline for the claim that some
stakeholders are highly visible in discourse but weaken as the corpus becomes
more institutional and more rule-bearing.

This script reports stakeholder prevalence at multiple stages:
1. Education chunks (discourse baseline)
2. Education-relevant institutional statements (sentence mentions)
3. Parsed A mentions in institutional statements
4. Explicit A mentions in institutional statements
5. Parsed A mentions in rules-only statements
6. Explicit A mentions in rules-only statements

It also computes within-IGT retention rates so I can distinguish: 
- discourse salience,
- institutionalization,
- explicit actor capture,
- rule-bearing survival.

Default inputs
--------------
- data/derived/step6_chunks_edu/chunks_edu.jsonl
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Default outputs
---------------
- data/derived/step8_analysis/stakeholder_attrition_master_edu/
    - stakeholder_stage_prevalence_long.csv
    - stakeholder_stage_prevalence_wide.csv
    - stakeholder_igt_retention_table.csv
    - stakeholder_stage_rankings.csv
    - stakeholder_attrition_summary.md
    - stakeholder_stage_prevalence_heatmap.png
    - stakeholder_igt_retention_heatmap.png
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )



def textify(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()



def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()



def compile_group_patterns(spec: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    return {
        group: [re.compile(pat, flags=re.IGNORECASE) for pat in pats]
        for group, pats in spec.items()
    }



def match_groups(text: str, compiled_patterns: Dict[str, List[re.Pattern]], ordered_groups: Sequence[str]) -> List[str]:
    s = textify(text).lower()
    if not s:
        return []
    hits: List[str] = []
    for group in ordered_groups:
        pats = compiled_patterns.get(group, [])
        if any(rx.search(s) for rx in pats):
            hits.append(group)
    return hits



def iter_chunks(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
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



def normalize_stmt_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()



def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0



def safe_ratio(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0



def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# Stakeholder groups (aligned to step8_5b)
# -----------------------------------------------------------------------------

ACTOR_GROUP_PATTERNS: Dict[str, List[str]] = {
    "educators_teachers": [
        r"\beducator(s)?\b",
        r"\bteacher(s)?\b",
        r"\binstructor(s)?\b",
        r"\blecturer(s)?\b",
        r"\btrainer(s)?\b",
        r"\bfaculty\b",
        r"\bprofessor(s)?\b",
        r"\bteaching staff\b",
    ],
    "students_learners": [
        r"\bstudent(s)?\b",
        r"\blearner(s)?\b",
        r"\bpupil(s)?\b",
        r"\bchild(ren)?\b",
        r"\bminor(s)?\b",
    ],
    "schools_institutions": [
        r"\bschool(s)?\b",
        r"\buniversity\b",
        r"\buniversities\b",
        r"\bcollege(s)?\b",
        r"\binstitution(s)?\b",
        r"\beducational institution(s)?\b",
        r"\bacademic institution(s)?\b",
        r"\btraining provider(s)?\b",
        r"\beducation provider(s)?\b",
    ],
    "policy_makers_ministries": [
        r"\bministry\b",
        r"\bministries\b",
        r"\bminister(s)?\b",
        r"\bgovernment\b",
        r"\bpublic authority\b",
        r"\bpublic authorities\b",
        r"\bpolicy maker(s)?\b",
        r"\bcompetent authority\b",
        r"\bcompetent authorities\b",
        r"\bdepartment of education\b",
        r"\beducation department\b",
        r"\bnational authority\b",
        r"\bstate\b",
    ],
    "regulators_supervisors": [
        r"\bregulator(s)?\b",
        r"\bsupervisory authority\b",
        r"\bsupervisory authorities\b",
        r"\boversight body\b",
        r"\boversight bodies\b",
        r"\benforcement authority\b",
        r"\bdata protection authority\b",
        r"\binspectorate\b",
        r"\bcommission\b",
        r"\bagency\b",
        r"\bboard\b",
        r"\bsupervisor(s)?\b",
    ],
    "platforms_systems": [
        r"\bsystem(s)?\b",
        r"\bplatform(s)?\b",
        r"\btool(s)?\b",
        r"\bmodel(s)?\b",
        r"\balgorithm(s)?\b",
        r"\bai system(s)?\b",
        r"\blearning management system(s)?\b",
        r"\blms\b",
        r"\bedtech\b",
    ],
    "commercial_designers_vendors": [
        r"\bvendor(s)?\b",
        r"\bprovider(s)?\b",
        r"\bsupplier(s)?\b",
        r"\bdeveloper(s)?\b",
        r"\bcompany\b",
        r"\bcompanies\b",
        r"\bmanufacturer(s)?\b",
        r"\bfirm(s)?\b",
        r"\bservice provider(s)?\b",
    ],
    "deployers_users_admins": [
        r"\bdeployer(s)?\b",
        r"\boperator(s)?\b",
        r"\buser(s)?\b",
        r"\badministrator(s)?\b",
        r"\bschool administrator(s)?\b",
        r"\bimplementer(s)?\b",
    ],
    "researchers_experts": [
        r"\bresearcher(s)?\b",
        r"\bscientist(s)?\b",
        r"\bexpert(s)?\b",
        r"\bacademic(s)?\b",
        r"\bresearch community\b",
        r"\bworking group\b",
    ],
    "parents_guardians_public": [
        r"\bparent(s)?\b",
        r"\bguardian(s)?\b",
        r"\bfamily\b",
        r"\bfamilies\b",
        r"\bcitizen(s)?\b",
        r"\bpublic\b",
        r"\bcivil society\b",
        r"\bngo(s)?\b",
    ],
}

ACTOR_GROUP_ORDER = list(ACTOR_GROUP_PATTERNS.keys())
DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str, fmt: str = ".2f") -> None:
    if df.empty:
        return
    data = df.to_numpy(dtype=float)
    fig_w = max(10, 1.2 * df.shape[1])
    fig_h = max(6, 0.55 * df.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Percent", rotation=270, labelpad=15)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks-jsonl",
        default="data/derived/step6_chunks_edu/chunks_edu.jsonl",
        help="Education chunk JSONL path",
    )
    ap.add_argument(
        "--igt-parquet",
        default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet",
        help="Education-relevant IGT parquet path",
    )
    ap.add_argument(
        "--out-dir",
        default="data/derived/step8_analysis/stakeholder_attrition_master_edu",
        help="Output directory",
    )
    ap.add_argument(
        "--rule-labels",
        nargs="*",
        default=sorted(DEFAULT_RULE_LABELS),
        help="Statement type labels treated as rules-only",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    chunks_path = Path(args.chunks_jsonl)
    igt_path = Path(args.igt_parquet)
    rule_labels = {str(x).strip().lower() for x in args.rule_labels}
    compiled = compile_group_patterns(ACTOR_GROUP_PATTERNS)

    chunks = iter_chunks(chunks_path)
    igt = maybe_read_table(igt_path)

    required_cols = {"sentence_text", "a_raw_text", "a_class", "statement_type_candidate"}
    missing = sorted(list(required_cols - set(igt.columns)))
    if missing:
        raise ValueError(f"Missing expected IGT columns: {missing}")

    igt = igt.copy()
    igt["_stmt_type_norm"] = normalize_stmt_type(igt["statement_type_candidate"])
    igt["_is_rule"] = igt["_stmt_type_norm"].isin(rule_labels)
    igt["_a_explicit"] = igt["a_class"].fillna("").astype(str).str.strip().str.lower().eq("explicit")

    # Match stakeholder groups across stages
    chunks["_chunk_groups"] = chunks["chunk_text"].apply(lambda x: match_groups(x, compiled, ACTOR_GROUP_ORDER))
    igt["_sent_groups"] = igt["sentence_text"].apply(lambda x: match_groups(x, compiled, ACTOR_GROUP_ORDER))
    igt["_a_groups"] = igt["a_raw_text"].apply(lambda x: match_groups(x, compiled, ACTOR_GROUP_ORDER))

    for grp in ACTOR_GROUP_ORDER:
        chunks[f"chunk_has_{grp}"] = chunks["_chunk_groups"].apply(lambda xs: grp in xs)
        igt[f"sent_has_{grp}"] = igt["_sent_groups"].apply(lambda xs: grp in xs)
        igt[f"a_has_{grp}"] = igt["_a_groups"].apply(lambda xs: grp in xs)
        igt[f"a_explicit_has_{grp}"] = igt[f"a_has_{grp}"] & igt["_a_explicit"]
        igt[f"rules_a_has_{grp}"] = igt[f"a_has_{grp}"] & igt["_is_rule"]
        igt[f"rules_a_explicit_has_{grp}"] = igt[f"a_has_{grp}"] & igt["_a_explicit"] & igt["_is_rule"]

    n_chunks = len(chunks)
    n_igt = len(igt)
    n_rules = int(igt["_is_rule"].sum())

    stage_specs = [
        ("education_chunks_any_mention", chunks, "chunk_has_{}", n_chunks),
        ("igt_sentence_mentions", igt, "sent_has_{}", n_igt),
        ("igt_a_mentions", igt, "a_has_{}", n_igt),
        ("igt_explicit_a_mentions", igt, "a_explicit_has_{}", n_igt),
        ("rules_a_mentions", igt, "rules_a_has_{}", n_igt),
        ("rules_explicit_a_mentions", igt, "rules_a_explicit_has_{}", n_igt),
    ]

    rows: List[dict] = []
    for stage_name, df_stage, col_tmpl, stage_total in stage_specs:
        for grp in ACTOR_GROUP_ORDER:
            col = col_tmpl.format(grp)
            hits = int(df_stage[col].sum())
            rows.append(
                {
                    "stakeholder_group": grp,
                    "stage": stage_name,
                    "hits": hits,
                    "stage_total_rows": int(stage_total),
                    "pct_stage_rows": pct(hits, stage_total),
                }
            )

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot(index="stakeholder_group", columns="stage", values="pct_stage_rows").fillna(0.0)
    wide_df = wide_df[[stage for stage, *_ in [(x[0],) for x in stage_specs]]]
    wide_df = wide_df.reset_index()

    # Rankings within stage
    ranking_df = long_df.sort_values(["stage", "pct_stage_rows", "hits"], ascending=[True, False, False]).copy()
    ranking_df["rank_within_stage"] = ranking_df.groupby("stage")["pct_stage_rows"].rank(method="dense", ascending=False).astype(int)

    # Within-IGT retention metrics
    retention_rows: List[dict] = []
    for grp in ACTOR_GROUP_ORDER:
        sent_n = int(igt[f"sent_has_{grp}"].sum())
        a_n = int(igt[f"a_has_{grp}"].sum())
        explicit_n = int(igt[f"a_explicit_has_{grp}"].sum())
        rule_a_n = int(igt[f"rules_a_has_{grp}"].sum())
        rule_explicit_n = int(igt[f"rules_a_explicit_has_{grp}"].sum())
        retention_rows.append(
            {
                "stakeholder_group": grp,
                "chunk_mentions_n": int(chunks[f"chunk_has_{grp}"].sum()),
                "igt_sentence_mentions_n": sent_n,
                "igt_a_mentions_n": a_n,
                "igt_explicit_a_mentions_n": explicit_n,
                "rules_a_mentions_n": rule_a_n,
                "rules_explicit_a_mentions_n": rule_explicit_n,
                "pct_sentence_to_a": safe_ratio(a_n, sent_n),
                "pct_a_to_explicit": safe_ratio(explicit_n, a_n),
                "pct_sentence_to_rules_explicit": safe_ratio(rule_explicit_n, sent_n),
                "pct_a_to_rules_explicit": safe_ratio(rule_explicit_n, a_n),
                "pct_explicit_to_rules_explicit": safe_ratio(rule_explicit_n, explicit_n),
            }
        )
    retention_df = pd.DataFrame(retention_rows).sort_values("igt_sentence_mentions_n", ascending=False)

    # Save tables
    path_long = out_dir / "stakeholder_stage_prevalence_long.csv"
    path_wide = out_dir / "stakeholder_stage_prevalence_wide.csv"
    path_ret = out_dir / "stakeholder_igt_retention_table.csv"
    path_rank = out_dir / "stakeholder_stage_rankings.csv"
    safe_to_csv(long_df, path_long)
    safe_to_csv(wide_df, path_wide)
    safe_to_csv(retention_df, path_ret)
    safe_to_csv(ranking_df, path_rank)

    # Heatmaps
    heat_stage = long_df.pivot(index="stakeholder_group", columns="stage", values="pct_stage_rows").fillna(0.0)
    heat_stage = heat_stage[[s[0] for s in stage_specs]]
    heat_ret = retention_df.set_index("stakeholder_group")[
        [
            "pct_sentence_to_a",
            "pct_a_to_explicit",
            "pct_sentence_to_rules_explicit",
            "pct_a_to_rules_explicit",
            "pct_explicit_to_rules_explicit",
        ]
    ]
    save_heatmap(heat_stage, out_dir / "stakeholder_stage_prevalence_heatmap.png", "Stakeholder prevalence by corpus stage")
    save_heatmap(heat_ret, out_dir / "stakeholder_igt_retention_heatmap.png", "Stakeholder retention within institutional statements")

    # Markdown summary
    top_chunks = long_df[long_df["stage"] == "education_chunks_any_mention"].sort_values("pct_stage_rows", ascending=False).head(8)
    top_rules = long_df[long_df["stage"] == "rules_explicit_a_mentions"].sort_values("pct_stage_rows", ascending=False).head(8)
    strongest_fall = retention_df.sort_values("pct_sentence_to_rules_explicit", ascending=True).head(8)

    md_lines = [
        "# Stakeholder attrition master table (education corpus)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Corpus sizes",
        f"- Education chunks: {n_chunks}",
        f"- Education-relevant institutional statements: {n_igt}",
        f"- Rules-only institutional statements: {n_rules}",
        "",
        "## Most visible stakeholders in education chunks",
        build_markdown_table(top_chunks[["stakeholder_group", "hits", "pct_stage_rows"]]),
        "",
        "## Most visible stakeholders in rules-only explicit-A statements",
        build_markdown_table(top_rules[["stakeholder_group", "hits", "pct_stage_rows"]]),
        "",
        "## Weakest sentence → rules-explicit retention",
        build_markdown_table(strongest_fall[["stakeholder_group", "igt_sentence_mentions_n", "rules_explicit_a_mentions_n", "pct_sentence_to_rules_explicit"]]),
        "",
        "Interpretation note: chunk-level visibility and sentence-level visibility are baselines; retention rates show whether actors survive parsing, explicit actor capture, and the rules-only restriction.",
    ]
    (out_dir / "stakeholder_attrition_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "chunks_jsonl": str(chunks_path),
            "igt_parquet": str(igt_path),
        },
        "outputs": {
            "stakeholder_stage_prevalence_long": str(path_long),
            "stakeholder_stage_prevalence_wide": str(path_wide),
            "stakeholder_igt_retention_table": str(path_ret),
            "stakeholder_stage_rankings": str(path_rank),
        },
        "rule_labels": sorted(rule_labels),
        "n_chunks": n_chunks,
        "n_igt_statements": n_igt,
        "n_rule_statements": n_rules,
        "stakeholder_groups": ACTOR_GROUP_ORDER,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [
        out_dir / "run_metadata.json",
        path_long,
        path_wide,
        path_ret,
        path_rank,
        out_dir / "stakeholder_attrition_summary.md",
        out_dir / "stakeholder_stage_prevalence_heatmap.png",
        out_dir / "stakeholder_igt_retention_heatmap.png",
    ]:
        print(f"  {p}")

    print("\nTop stakeholders in education chunks:")
    print(top_chunks[["stakeholder_group", "hits", "pct_stage_rows"]].to_string(index=False))
    print("\nTop stakeholders in rules-only explicit-A statements:")
    print(top_rules[["stakeholder_group", "hits", "pct_stage_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
