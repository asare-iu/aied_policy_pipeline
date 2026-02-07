#!/usr/bin/env python3
"""
Step 5.5 — Variance explainability for full-corpus 40PC model.

Outputs:
  - variance_by_pc.csv
  - variance_by_admissibility.csv
  - cumulative_variance.csv
  - cumulative_variance.png

Assumptions:
  - Model dir contains a fitted PCA/TruncatedSVD object saved via joblib/pickle,
    OR a metadata json that already includes explained variance ratio.
  - Human_interpretation.csv exists and has columns: pc, admissibility, tags, short_label, long_label
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# matplotlib only for file output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_variance_from_metadata(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Try a few common keys
    for key in ["explained_variance_ratio", "explained_variance_ratio_", "variance_ratio", "evr"]:
        if key in meta and isinstance(meta[key], list) and len(meta[key]) > 0:
            return meta[key]

    # Sometimes stored under nested dict
    for container_key in ["pca", "svd", "model"]:
        if container_key in meta and isinstance(meta[container_key], dict):
            for key in ["explained_variance_ratio", "explained_variance_ratio_", "variance_ratio", "evr"]:
                v = meta[container_key].get(key)
                if isinstance(v, list) and len(v) > 0:
                    return v

    return None


def load_model_object(model_dir: Path):
    """
    Heuristic: look for a joblib/pkl containing the PCA/SVD object.
    We choose the first candidate that exposes explained_variance_ratio_.
    """
    try:
        import joblib
    except ImportError:
        raise RuntimeError("joblib is required but not installed in this environment.")

    candidates = []
    # common naming patterns
    candidates += sorted(model_dir.glob("*.joblib"))
    candidates += sorted(model_dir.glob("*.pkl"))
    candidates += sorted(model_dir.glob("*.pickle"))

    if not candidates:
        raise FileNotFoundError(f"No joblib/pkl/pickle files found in {model_dir}")

    for path in candidates:
        try:
            obj = joblib.load(path)
        except Exception:
            continue

        # sometimes saved as a dict bundle
        if isinstance(obj, dict):
            for k, v in obj.items():
                if hasattr(v, "explained_variance_ratio_"):
                    return v, path
        else:
            if hasattr(obj, "explained_variance_ratio_"):
                return obj, path

    raise RuntimeError(
        "Could not find a PCA/TruncatedSVD object with explained_variance_ratio_ in model_dir. "
        "If you saved metadata JSON with EVR, point --metadata_json to it."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        default="data/derived/step5_models_full_40pc",
        help="Directory containing the fitted full-corpus 40PC model artifacts.",
    )
    ap.add_argument(
        "--human_csv",
        default="data/derived/step5_models_full_artifact_stripped_40pc/Human_interpretation.csv",
        help="Human interpretation CSV with admissibility + tags.",
    )
    ap.add_argument(
        "--metadata_json",
        default="",
        help="Optional explicit path to metadata JSON containing explained variance ratio.",
    )
    ap.add_argument(
        "--out_dir",
        default="data/derived/step5_models_full_40pc",
        help="Output directory for variance explainability artifacts.",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1) Get explained variance ratio list (length = n_components)
    # ----------------------------
    evr = None

    # Explicit metadata path
    if args.metadata_json:
        meta_path = Path(args.metadata_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"--metadata_json not found: {meta_path}")
        evr = load_variance_from_metadata(meta_path)

    # Otherwise try common metadata names in model_dir
    if evr is None:
        meta_guess = find_first_existing([
            model_dir / "pca_full_metadata.json",
            model_dir / "pca_metadata.json",
            model_dir / "metadata.json",
            model_dir / "model_metadata.json",
        ])
        if meta_guess:
            evr = load_variance_from_metadata(meta_guess)

    # Fallback: load model object
    if evr is None:
        model_obj, model_path = load_model_object(model_dir)
        evr = list(getattr(model_obj, "explained_variance_ratio_"))
        print(f"Loaded model object from: {model_path}")

    if not evr or len(evr) == 0:
        raise RuntimeError("Explained variance ratio could not be loaded from metadata or model object.")

    # ----------------------------
    # 2) Build variance_by_pc table
    # ----------------------------
    pcs = [f"PC{i}" for i in range(1, len(evr) + 1)]
    var_df = pd.DataFrame({
        "pc": pcs,
        "pc_num": list(range(1, len(evr) + 1)),
        "explained_variance_ratio": evr,
    })
    var_df["cumulative_variance_ratio"] = var_df["explained_variance_ratio"].cumsum()

    # ----------------------------
    # 3) Join human interpretation (admissibility + tags)
    # ----------------------------
    human_path = Path(args.human_csv)
    if not human_path.exists():
        raise FileNotFoundError(f"Human interpretation CSV not found: {human_path}")

    human_df = pd.read_csv(human_path)

    required = {"pc", "admissibility", "tags"}
    missing = required - set(human_df.columns)
    if missing:
        raise ValueError(f"Human CSV missing required columns: {sorted(missing)}")

    merged = var_df.merge(human_df, on="pc", how="left")

    # sanity: if many NaNs, the PC naming differs
    if merged["admissibility"].isna().any():
        missing_pcs = merged[merged["admissibility"].isna()]["pc"].tolist()
        print(f"WARNING: Missing admissibility for PCs: {missing_pcs[:10]}{'...' if len(missing_pcs)>10 else ''}")
        # keep them but label unknown
        merged["admissibility"] = merged["admissibility"].fillna("unknown")
        merged["tags"] = merged["tags"].fillna("admissibility_unknown")

    # Write variance_by_pc.csv (full join)
    out_variance_by_pc = out_dir / "variance_by_pc.csv"
    merged.drop(columns=["pc_num"]).to_csv(out_variance_by_pc, index=False)

    # ----------------------------
    # 4) Group variance by admissibility
    # ----------------------------
    group_df = (
        merged.groupby("admissibility", as_index=False)["explained_variance_ratio"]
        .sum()
        .rename(columns={"explained_variance_ratio": "total_explained_variance_ratio"})
        .sort_values("total_explained_variance_ratio", ascending=False)
    )
    out_group = out_dir / "variance_by_admissibility.csv"
    group_df.to_csv(out_group, index=False)

    # ----------------------------
    # 5) Cumulative variance table + plot
    # ----------------------------
    cum_df = merged[["pc", "explained_variance_ratio", "cumulative_variance_ratio", "admissibility"]].copy()
    out_cum = out_dir / "cumulative_variance.csv"
    cum_df.to_csv(out_cum, index=False)

    # Plot cumulative variance
    plt.figure()
    plt.plot(list(range(1, len(evr) + 1)), var_df["cumulative_variance_ratio"])
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Full-Corpus PCA (40PC): Cumulative Explained Variance")
    plt.grid(True, linewidth=0.5)
    out_png = out_dir / "cumulative_variance.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Print brief summary to terminal
    total = float(var_df["explained_variance_ratio"].sum())
    print(f"Wrote: {out_variance_by_pc}")
    print(f"Wrote: {out_group}")
    print(f"Wrote: {out_cum}")
    print(f"Wrote: {out_png}")
    print(f"Total explained variance ratio (sum of PCs): {total:.4f}")


if __name__ == "__main__":
    main()
