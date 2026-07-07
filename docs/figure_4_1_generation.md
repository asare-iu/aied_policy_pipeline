# Figure 4.1 generation

This note documents how to generate the dissertation-style Figure 4.1 used in Chapter 4.

## Project root

The VM inventory shows the repository root as:

```bash
/home/isakntiasare/aied_policy_pipeline
```

The source code also assumes this root. Several scripts define the project root as the parent of the `scripts/` directory or use paths relative to the repository root. Data products are stored under `data/derived/`.

For Figure 4.1, the default input file is:

```bash
data/derived/step5_5_pca_interpretation/pca_pc_labels_final.csv
```

The figure script writes outputs to:

```bash
figures/chapter4/
```

## Run instructions

From the VM terminal:

```bash
cd /home/isakntiasare/aied_policy_pipeline
python3 -m pip install pandas matplotlib
python3 scripts/step5_9_make_substantive_component_chart.py
```

The script creates:

```bash
figures/chapter4/figure_4_1_substantive_governance_components.png
figures/chapter4/figure_4_1_substantive_governance_components.pdf
figures/chapter4/figure_4_1_substantive_governance_components.svg
```

Use the PNG for Word or Google Docs. Use the PDF for LaTeX or other workflows that support vector-quality figures. Use the SVG for editing in Inkscape, Illustrator, or similar tools.

## Optional arguments

To pass a different labels CSV:

```bash
python3 scripts/step5_9_make_substantive_component_chart.py \
  --labels-csv path/to/pca_pc_labels_final.csv
```

To write to a different output directory:

```bash
python3 scripts/step5_9_make_substantive_component_chart.py \
  --output-dir figures/chapter4
```

## Dissertation caption

Do not put the formal caption inside the figure image. Add it below the figure in the dissertation manuscript:

> **Figure 4.1. Substantive governance components ranked by principal component number.** The chart shows the 16 components classified as substantive among the first 25 interpreted components in the full-corpus TruncatedSVD solution. Education-relevant substantive components are highlighted.  
> *Source:* `pca_pc_labels_final.csv`.
