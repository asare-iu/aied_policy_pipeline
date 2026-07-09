# Chapter 4 structural visuals manifest

## Source files
- full: `/home/isakntiasare/aied_policy_pipeline/data/derived/step5_models_full_40pc/row_alignment.csv`
- education: `/home/isakntiasare/aied_policy_pipeline/data/derived/step5_models_edu_embedded_40pc/row_alignment.csv`
- title_clean: `/home/isakntiasare/aied_policy_pipeline/data/derived/step5_models_title_edu_40pc_clean/row_alignment.csv`
- full_country: `/home/isakntiasare/aied_policy_pipeline/data/derived/step9_country_dataset/country_governance_dataset.csv`
- edu_country: `/home/isakntiasare/aied_policy_pipeline/data/derived/step10_education_dataset/education_country_dataset.csv`
- cluster_profiles: `/home/isakntiasare/aied_policy_pipeline/data/derived/step10_education_dataset/qc/education_cluster_profiles.csv`

## Key computed values
- Full chunk total: 47,512
- Education content-gated chunks: 10,557 (22.2% of full corpus)
- Clean title-gated chunks: 1,795 (3.8% of full corpus)
- Direct education-relevant chunks, chunk gate and title gate: 703 (6.7% of education-relevant chunks)
- Cross-domain education-relevant chunks, chunk gate outside title gate: 9,854 (93.3% of education-relevant chunks)
- Full corpus top-three statement share: 39.3%
- Education-relevant top-three statement share: 44.1%
- Full median statements per document: 110.8
- Education-relevant median statements per document: 9.3

## Figure captions
- Figure 4.4. Two-gate structure of education-relevant AI governance. The figure cross-classifies chunks by the chunk-level education content gate and the cleaned document-title education gate.
- Figure 4.5. Cumulative concentration of institutional statements by jurisdiction. Jurisdictions are ranked by statement volume for the full corpus and the education-relevant subset.
- Figure 4.6. Jurisdictional rank shifts from overall AI governance to education-relevant AI governance. The education-relevant subset changes the hierarchy while retaining a concentrated head.
- Figure 4.7. Institutional density cliff. Country-level statements per document are substantially lower in the education-relevant subset than in the full corpus.
- Figure 4.8. Cluster constellation of education-relevant AI governance profiles. Bubble size represents the number of jurisdictions in each cluster.
