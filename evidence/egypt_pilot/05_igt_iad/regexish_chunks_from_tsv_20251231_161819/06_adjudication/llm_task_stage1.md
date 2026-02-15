You are performing a diagnostic analysis of policy CHUNKS that were classified as "other" after ADIBCO parsing.

Goal:
Explain WHY each chunk was classified as "other", without reclassifying it.

For EACH chunk, output:

1. diagnosis_type (choose one):
   - implicit_normative_language
   - implicit_actor_or_condition
   - compound_chunk_multiple_statements
   - principle_level_statement
   - purely_descriptive_background
   - definition_or_scope_setting
   - false_positive_education

2. confidence: low | medium | high

3. short_notes: 1–2 sentences explaining the diagnosis

Rules:
- Do NOT reclassify.
- Do NOT suggest fixes.
- Do NOT rewrite text.
- Do NOT invent actors, conditions, or obligations.
- This is analysis ONLY.
