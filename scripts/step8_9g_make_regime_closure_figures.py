#!/usr/bin/env python3
"""
Step 8.9G — Generate visuals for edu regime closure + enforcement attachment

Reads:
- data/derived/step8_9_regime_closure/edu_closure_doc_summary_refined.csv
- data/derived/step8_9_regime_closure/reports/edu_closure_enforcement_attachment.csv
- data/derived/step8_9_regime_closure/reports/edu_direct_vs_indirect_empowerment_with_enforcement.csv

Writes (PNG + PDF):
- data/derived/step8_9_regime_closure/figures/*.png
- data/derived/step8_9_regime_closure/figures/*.pdf

Notes:
- matplotlib only (no seaborn)
- No explicit colors set (defaults only)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED = PROJECT_ROOT / "data" / "derived" / "step8_9_regime_closure"
REPORTS = DERIVED / "reports"
OUTDIR = DERIVED / "figures"


def _ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, stem: str) -> None:
    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[fig] wrote {png}")
    print(f"[fig] wrote {pdf}")


def fig1_closure_method_distribution(doc_summary: pd.DataFrame) -> None:
    """
    Bar chart: count of documents by closure_method.
    """
    vc = doc_summary["closure_method"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("Education Regime Closure: Documents by Closure Method")
    ax.set_xlabel("Closure method")
    ax.set_ylabel("Number of documents")
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(vc.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    _save(fig, "fig1_closure_method_distribution")


def fig2_closure_size_vs_strict_enforcement(enf: pd.DataFrame) -> None:
    """
    Scatter: closure_size vs pct_o_umbrella_linked_strict, by method.
    """
    df = enf.copy()
    for c in ["closure_size", "pct_o_umbrella_linked_strict"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["closure_size", "pct_o_umbrella_linked_strict"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Plot each method as its own series (matplotlib auto-cycles styles)
    for m, g in df.groupby("method"):
        ax.scatter(g["closure_size"], g["pct_o_umbrella_linked_strict"], label=str(m), alpha=0.7)

    ax.set_title("Closure Size vs Strict Umbrella Enforcement Attachment")
    ax.set_xlabel("Closure size (number of closure-linked statements)")
    ax.set_ylabel("Strict umbrella linkage (%)")
    ax.legend(title="Closure method", loc="best", fontsize=9)
    _save(fig, "fig2_closure_size_vs_strict_enforcement")


def fig3_direct_vs_indirect_empowerment_stacked(emp: pd.DataFrame) -> None:
    """
    Stacked bars: pct_directA vs pct_indirect vs (100 - direct - indirect)
    plus overlay marker for pct_o_umbrella_strict.
    """
    df = emp.copy()

    # ensure numeric
    for c in ["pct_directA", "pct_indirect", "pct_o_umbrella_strict"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("statement_type_candidate")

    labels = df["statement_type_candidate"].astype(str).tolist()
    direct = df["pct_directA"].to_numpy()
    indirect = df["pct_indirect"].to_numpy()
    remainder = np.clip(100.0 - direct - indirect, 0.0, 100.0)
    strict = df["pct_o_umbrella_strict"].to_numpy()

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(x, direct, label="Direct education Actor A (%)")
    ax.bar(x, indirect, bottom=direct, label="Indirect education inclusion (%)")
    ax.bar(x, remainder, bottom=direct + indirect, label="Other / not education-positioned (%)")

    # Overlay strict umbrella attachment as markers
    ax.plot(x, strict, marker="o", linestyle="-", label="Strict umbrella enforcement (%)")

    ax.set_title("Education Institutional Positioning and Enforcement Attachment")
    ax.set_xlabel("Statement type")
    ax.set_ylabel("Percent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", fontsize=9)

    # annotate key values
    for i in range(len(labels)):
        ax.text(x[i], direct[i], f"{direct[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i], direct[i] + indirect[i], f"{indirect[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i], strict[i], f"{strict[i]:.2f}", ha="center", va="bottom", fontsize=8)

    _save(fig, "fig3_direct_vs_indirect_empowerment_stacked")


def fig4_enforcement_type_heatmap(enf: pd.DataFrame) -> None:
    """
    Heatmap: by closure method (rows) x enforcement measures (cols)
    using mean percentages across docs.
    """
    df = enf.copy()
    # numeric conversions
    cols = ["pct_o_local_present", "pct_o_umbrella_present", "pct_o_umbrella_linked_strict", "pct_umbrella_doc_fallback"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (
        df.groupby("method")[cols]
        .mean(numeric_only=True)
        .sort_values("pct_o_umbrella_linked_strict", ascending=False)
    )

    data = agg.to_numpy()
    row_labels = agg.index.astype(str).tolist()
    col_labels = [
        "% local O present",
        "% umbrella present",
        "% strict umbrella linked",
        "% umbrella doc fallback",
    ]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    im = ax.imshow(data, aspect="auto")

    ax.set_title("Enforcement Attachment Patterns by Closure Method (Mean % across documents)")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")

    # annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, "fig4_enforcement_type_heatmap")


def fig5_regime_architecture_diagram() -> None:
    """
    Conceptual figure: Regime closure model diagram (E -> closure -> R -> enforcement).
    """
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.axis("off")

    # Box positions
    boxes = {
        "E": (0.05, 0.55, 0.24, 0.30),
        "C": (0.36, 0.55, 0.28, 0.30),
        "R": (0.70, 0.55, 0.24, 0.30),
        "O": (0.36, 0.10, 0.58, 0.30),
    }

    def box(key, title, lines):
        x, y, w, h = boxes[key]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False))
        ax.text(x + 0.01, y + h - 0.06, title, fontsize=12, fontweight="bold", va="top")
        ax.text(x + 0.01, y + h - 0.12, "\n".join(lines), fontsize=10, va="top")

    box(
        "E",
        "Education signal (E)",
        [
            "Domain terms (education/training/schools)",
            "Education actors (ministries, schools, teachers, etc.)",
        ],
    )
    box(
        "C",
        "Closure mechanism",
        [
            "Anchor overlap (condition-anchored)",
            "Global scope clause",
            "Semantic similarity fallback",
        ],
    )
    box(
        "R",
        "Regime rule set (R)",
        [
            "Rule candidates + norm candidates",
            "Closure-linked statements (doc-level regime)",
        ],
    )
    box(
        "O",
        "Enforcement attachment (O)",
        [
            "Local O: statement-level sanction clauses",
            "Umbrella O: document/section sanction blocks",
            "Strict linkage vs doc fallback inheritance",
        ],
    )

    # arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))

    # E -> C -> R
    arrow(0.29, 0.70, 0.36, 0.70)
    arrow(0.64, 0.70, 0.70, 0.70)
    # R -> O (down)
    arrow(0.82, 0.55, 0.82, 0.40)
    # C -> O (down)
    arrow(0.50, 0.55, 0.50, 0.40)

    ax.text(0.36, 0.92, "Regime Closure Model for Education in AI Governance Texts", fontsize=13, fontweight="bold")

    _save(fig, "fig5_regime_architecture_diagram")


def fig6_actorA_presence_vs_strict_enforcement(doc_summary: pd.DataFrame, enf: pd.DataFrame) -> None:
    """
    Two-bar comparison:
    docs with education Actor A present vs absent, comparing mean strict umbrella linkage.
    """
    ds = doc_summary[["doc_id", "doc_has_edu_actor_A"]].copy()
    ds["doc_has_edu_actor_A"] = ds["doc_has_edu_actor_A"].fillna(False).astype(bool)

    e = enf[["doc_id", "pct_o_umbrella_linked_strict"]].copy()
    e["pct_o_umbrella_linked_strict"] = pd.to_numeric(e["pct_o_umbrella_linked_strict"], errors="coerce")

    m = ds.merge(e, on="doc_id", how="inner").dropna(subset=["pct_o_umbrella_linked_strict"])

    grp = m.groupby("doc_has_edu_actor_A")["pct_o_umbrella_linked_strict"].mean()
    # Ensure both keys exist
    mean_false = float(grp.get(False, np.nan))
    mean_true = float(grp.get(True, np.nan))

    labels = ["No edu Actor A", "Edu Actor A present"]
    vals = [mean_false, mean_true]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, vals)
    ax.set_title("Strict Enforcement Attachment by Presence of Education Actor A (Doc-level)")
    ax.set_ylabel("Mean strict umbrella linkage (%)")
    ax.set_ylim(0, max([v for v in vals if not np.isnan(v)] + [1.0]) * 1.25)

    for i, v in enumerate(vals):
        if np.isnan(v):
            txt = "NA"
            y = 0
        else:
            txt = f"{v:.2f}"
            y = v
        ax.text(i, y, txt, ha="center", va="bottom", fontsize=10)

    _save(fig, "fig6_actorA_presence_vs_strict_enforcement")


def main() -> None:
    t0 = time.time()
    _ensure_outdir()

    p_doc_summary = DERIVED / "edu_closure_doc_summary_refined.csv"
    p_enf = REPORTS / "edu_closure_enforcement_attachment.csv"
    p_emp = REPORTS / "edu_direct_vs_indirect_empowerment_with_enforcement.csv"

    for p in [p_doc_summary, p_enf, p_emp]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    doc_summary = pd.read_csv(p_doc_summary)
    enf = pd.read_csv(p_enf)
    emp = pd.read_csv(p_emp)

    print("[step8_9g] loaded:")
    print(" -", p_doc_summary, "rows=", len(doc_summary))
    print(" -", p_enf, "rows=", len(enf))
    print(" -", p_emp, "rows=", len(emp))

    fig1_closure_method_distribution(doc_summary)
    fig2_closure_size_vs_strict_enforcement(enf)
    fig3_direct_vs_indirect_empowerment_stacked(emp)
    fig4_enforcement_type_heatmap(enf)
    fig5_regime_architecture_diagram()
    fig6_actorA_presence_vs_strict_enforcement(doc_summary, enf)

    print(f"[step8_9g] done elapsed_s={time.time() - t0:.1f}")
    print(f"[step8_9g] outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
