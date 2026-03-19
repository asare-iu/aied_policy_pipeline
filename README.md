# AIED Policy Pipeline

A research pipeline for building, filtering, annotating, and analyzing a global corpus of artificial intelligence policy documents, with a particular focus on how education is represented within AI governance discourse.

This repository supports a multi-stage workflow that moves from raw document ingestion and text normalization through chunking, embedding, dimensionality reduction, institutional statement extraction, rule classification, country-level aggregation, and targeted education-specific analyses. The pipeline is designed to support dissertation research on the governance of Artificial Intelligence in Education (AIED), especially through institutional and policy-analytic lenses.

## Purpose

The repository serves three related purposes:

1. **Corpus construction and preparation**  
   It organizes and processes a cross-national corpus of AI policy documents, including raw document inventories, normalized text, chunked text, and multiple education-relevant subsets.

2. **Computational policy analysis**  
   It supports computational analysis of the corpus through embeddings, PCA, clustering, institutional grammar-inspired parsing, and rule-type classification.

3. **Methodological transparency and auditability**  
   It preserves selected derived outputs, human-coded artifacts, adjudication sheets, and summary tables so that the analytical workflow is inspectable and reproducible at a meaningful level.

## Research context

This repository underpins research on how AI policy frameworks govern, imagine, and structure education. It is especially concerned with questions such as:

- how education is represented in national and international AI policy
- how institutional statements about education can be extracted and classified
- how countries differ in the balance between strategic, regulatory, informational, and governance-oriented approaches
- how education-specific governance can be studied using institutional analysis and development-inspired concepts

The workflow is not just a generic NLP pipeline. It is a research pipeline tailored to policy analysis, institutional grammar, and education governance.

## High-level pipeline structure

The exact scripts and outputs evolve over time, but the workflow broadly follows the structure below.

### Step 1: Text extraction and normalization
Raw policy documents are converted into normalized machine-readable text. This stage produces cleaned document text and basic diagnostics about extraction quality.

Representative outputs include:
- normalized document text
- extraction error summaries
- document inventories

### Step 2: Chunking
Normalized documents are segmented into chunks for downstream embedding and analysis.

Representative outputs include:
- chunk-level text outputs
- chunk count summaries by document

### Step 3–5: Embeddings, dimensionality reduction, and interpretive modeling
The pipeline generates text embeddings and then applies PCA-based dimensionality reduction and interpretation workflows across the full corpus and selected subsets.

Representative outputs include:
- PCA explained variance tables
- top-term summaries for components
- PCA interpretation sheets
- validation summaries
- full and subset score outputs

### Step 6–7: Education-focused subset construction
Education-relevant and education-in-title subsets are constructed, audited, and compared.

Representative outputs include:
- education subset summaries
- country dispersion summaries
- document dispersion summaries
- manual exclusions for title-based education subsets

### Step 8: Institutional statement extraction and rule analysis
This stage identifies and classifies institutional statements and rule-like expressions using an expanded IGT / institutional grammar-inspired logic. It also includes multiple cleaning, QC, and adjudication workflows.

Representative outputs include:
- IGT statement tables
- rule candidate tables
- rule-type summaries
- adjudication sheets
- manually coded methodological artifacts
- education-relevant rule subsets
- regime closure and umbrella scope analyses

### Step 9: Country-level governance aggregation
Step 9 aggregates institutional and governance features to the country level and generates country-level governance datasets, PCA coordinates, clusters, figures, and tables.

Representative outputs include:
- `country_governance_dataset.csv`
- `country_governance_indices.csv`
- `country_pca_coordinates.csv`
- `country_cluster_assignments.csv`
- country-level governance figures
- cluster summary tables

### Step 10 and beyond
Later stages extend the workflow into additional education-layer analyses and downstream interpretation. These stages may still be in development or under active revision depending on the branch and current phase of the research.

## Repository structure

The repository is organized around scripts, configuration, and staged derived outputs.

### `scripts/`
Contains the executable pipeline scripts for individual stages. These are generally named by pipeline step (for example `step9_2_build_country_governance_dataset.py`).

### `data/raw/`
Contains raw or near-raw source materials and supporting corpora where appropriate.

### `data/derived/`
Contains selected derived outputs from different pipeline stages. Some derived outputs are committed for transparency and reuse; others are intentionally ignored because they are too large, too intermediate, or too easily regenerated.

### `config/`
Contains configuration files, support lexicons, and other pipeline settings.

## What is committed to the repository

This repository includes a curated subset of derived artifacts. In general, the public repository prioritizes:

- scripts required to understand and reproduce the workflow
- compact derived CSVs that support interpretation and downstream analysis
- final or near-final figures and tables
- selected human-coded and adjudication artifacts
- appendix-supporting inventories and readable summaries

Examples of committed artifact types include:
- PCA explained variance outputs
- PCA top-term summaries
- education subset summaries
- Step 9 country governance figures and tables
- manual exclusions used in subset construction
- rule manual coding sheets
- QC adjudication sheets
- human-readable appendix inventory outputs

## What may be intentionally omitted

Not every generated file is committed. Some categories of outputs are often kept local or excluded from version control, including:

- large intermediate parquet files
- bulky row-level matrices that can be regenerated
- temporary audit bundles
- local environment inventories
- exploratory exports not needed for long-term interpretation
- generated text artifacts that are too large or too mechanical to add value in the public repo

This is a deliberate design choice. The goal is to make the workflow inspectable without turning the repository into a dump of every transient artifact ever produced.

## Human-coded and methodological audit artifacts

A central goal of this repository is methodological transparency. For that reason, selected human-coded and manually curated files are retained where they substantively document how computational decisions were checked, refined, or adjudicated.

Examples include:
- PCA label interpretation sheets
- manual exclusions for education-title subset construction
- rule manual coding sheets
- QC adjudication sheets and summaries

These artifacts are important because they document the interaction between automated processing and human review.

## Step 9 country governance outputs

A major analytical layer in the repository is the country-level governance dataset produced in Step 9. This part of the workflow:
- attaches countries to institutional statements
- constructs country-level governance indicators
- applies PCA
- generates cluster assignments
- produces figures and cluster tables

Representative Step 9 scripts include:
- `scripts/step9_0_attach_country.py`
- `scripts/step9_1_attach_country_to_igt.py`
- `scripts/step9_2_build_country_governance_dataset.py`
- `scripts/step9_3_governance_indices_pca_clustering.py`
- `scripts/step9_4_generate_governance_figures_tables.py`

Representative Step 9 outputs include:
- `data/derived/step9_country_dataset/country_governance_dataset.csv`
- `data/derived/step9_country_dataset/country_governance_indices.csv`
- `data/derived/step9_country_dataset/country_pca_coordinates.csv`
- `data/derived/step9_country_dataset/country_cluster_assignments.csv`
- `data/derived/step9_country_dataset/figures/`
- `data/derived/step9_country_dataset/tables/`

## Appendix outputs

The repository also includes appendix-oriented outputs that make the corpus easier to inspect.

Examples include:
- `data/derived/appendix_outputs/AI_Policy_Document_Inventory_Appendix.csv`
- `data/derived/appendix_outputs/AI_Policy_Document_Inventory_Readable.txt`

These files provide readable summaries of corpus contents and support transparent reporting.

## Notes on reproducibility

This repository is best understood as a **research-grade reproducibility repository**, not a one-click packaged software library.

That means:
- some steps depend on local environment setup
- some raw inputs may require separate preparation
- some large intermediates may be omitted from git
- some scripts assume the presence of earlier outputs in expected directories

The repository is still intended to make the analytical logic, data flow, and key outputs auditable and reusable.

## Recommended way to use this repository

A reader or collaborator will get the most value by approaching the repository in this order:

1. Read this README for the overall logic.
2. Inspect `scripts/` to understand the pipeline stages.
3. Review committed derived outputs in `data/derived/`.
4. Inspect human-coded and adjudication artifacts for methodological transparency.
5. Focus on Step 9 and appendix outputs for high-level interpretive products.

## Current status

This repository is under active development. Some outputs are stable and interpretive; others are interim artifacts retained for transparency and may later be pruned, reorganized, or replaced by more final versions.

Where intermediate derived artifacts are temporarily included, they are present to support auditability and methodological review.

## Suggested future improvements

Planned or useful future improvements may include:
- a fuller environment/setup guide
- a step-by-step reproduction order
- explicit input/output manifests for each stage
- a data retention policy describing what classes of derived artifacts are committed
- a methods note explaining the institutional grammar and rule-classification logic in greater detail

## License and use

Unless otherwise specified, users should treat this repository as research code and research data infrastructure. Please review the repository contents carefully before reusing outputs, especially where raw policy sources, translated materials, or manually curated research artifacts are involved.

## Contact / authorship

This repository supports dissertation and research work by Isak Nti Asare on the governance of AI in education and related policy analysis questions.

If you are using the repository for collaboration, interpretation, or extension, start by reviewing the scripts and committed derived outputs most relevant to your part of the workflow.
