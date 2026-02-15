#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_models_outputs(models_dir: Path) -> Dict[str, Path]:
    """
    Returns paths to expected PCA artifacts produced by step5_0_fit_full_models.py
    """
    models_dir = Path(models_dir)
    paths = {
        "explained_variance": models_dir / "explained_variance.csv",
        "scores": models_dir / "full_scores_from_models.csv",
        "top_terms": models_dir / "top_terms.csv",
        "top_terms_long": models_dir / "top_terms_long.csv",
        "pc_interpretations": models_dir / "pc_interpretations.csv",
    }
    return paths


def load_terms_long(p: Path) -> pd.DataFrame:
    """
    Expects columns: pc, side (pos/neg), rank, term, weight
    """
    df = pd.read_csv(p)
    needed = {"pc", "side", "rank", "term"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"top_terms_long.csv missing columns: {sorted(missing)}")
    return df


def load_pc_interpretations(p: Path) -> pd.DataFrame:
    """
    Expects columns: pc, interpretable (Y/N), admissible (Y/N), label_expanded, note, ...
    """
    df = pd.read_csv(p)
    if "pc" not in df.columns:
        raise ValueError("pc_interpretations.csv must contain a 'pc' column")
    for col in ["admissible", "interpretable"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def get_interpretable_pcs(df_interp: pd.DataFrame, fallback_n: int = 3) -> List[str]:
    """
    Return list of PCs marked interpretable=='Y'. If none, fallback to first N PCs.
    """
    if "interpretable" in df_interp.columns:
        pcs = df_interp.loc[df_interp["interpretable"] == "Y", "pc"].astype(str).tolist()
        if pcs:
            return pcs
    return [f"PC{i}" for i in range(1, fallback_n + 1)]


def plot_explained_variance(df_var: pd.DataFrame, out_path: Path) -> None:
    """
    df_var columns: pc, explained_variance_ratio, cumulative
    """
    _ensure_dir(Path(out_path).parent)

    # Ensure order PC1..PCn
    pcs = df_var["pc"].astype(str).tolist()
    x = np.arange(1, len(pcs) + 1)
    y = df_var["explained_variance_ratio"].astype(float).values
    cum = df_var["cumulative"].astype(float).values

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained variance ratio")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # cumulative
    out_cum = out_path.with_name(out_path.stem.replace("explained_variance", "cumulative_variance") + out_path.suffix)
    plt.figure()
    plt.plot(x, cum, marker="o")
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative explained variance")
    plt.tight_layout()
    plt.savefig(out_cum, dpi=200)
    plt.close()


def plot_scores_scatter(df_scores: pd.DataFrame, out_path: Path, pc_x: str, pc_y: str) -> None:
    """
    df_scores expected to contain columns named like PC1, PC2, ...
    """
    _ensure_dir(Path(out_path).parent)
    if pc_x not in df_scores.columns or pc_y not in df_scores.columns:
        raise ValueError(f"Scores missing {pc_x} and/or {pc_y}. Available: {list(df_scores.columns)[:20]} ...")

    plt.figure()
    plt.scatter(df_scores[pc_x].values, df_scores[pc_y].values, s=8, alpha=0.6)
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loadings_bar(df_terms_long: pd.DataFrame, out_path: Path, pc: str, top_k: int = 20) -> None:
    """
    Bar plot of top positive/negative terms for a PC from df_terms_long.
    """
    _ensure_dir(Path(out_path).parent)
    sub = df_terms_long[df_terms_long["pc"].astype(str) == pc].copy()
    if sub.empty:
        raise ValueError(f"No rows for {pc} in top_terms_long")

    # Expect 'side' in {pos,neg} and 'rank' ascending
    pos = sub[sub["side"].astype(str).str.lower() == "pos"].sort_values("rank").head(top_k)
    neg = sub[sub["side"].astype(str).str.lower() == "neg"].sort_values("rank").head(top_k)

    # Use weights if available; else use descending rank proxy
    if "weight" in sub.columns:
        pos_vals = pos["weight"].astype(float).values
        neg_vals = neg["weight"].astype(float).values
    else:
        pos_vals = np.arange(len(pos), 0, -1)
        neg_vals = -np.arange(len(neg), 0, -1)

    terms = list(reversed(neg["term"].astype(str).tolist())) + pos["term"].astype(str).tolist()
    vals = list(reversed(neg_vals.tolist())) + pos_vals.tolist()

    plt.figure(figsize=(10, 6))
    y = np.arange(len(terms))
    plt.barh(y, vals)
    plt.yticks(y, terms)
    plt.xlabel("Loading weight (or rank proxy)")
    plt.title(pc)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
