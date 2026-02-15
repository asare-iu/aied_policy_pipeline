TASK (senior-coder pass on "other" chunks):
You are reviewing chunks that were classified as "other" in Stage 1.

Given:
(1) chunk_text (source of truth),
(2) Stage 1 metadata (baseline_statement_type, stage1_final_statement_type=other, bucket, evidence, missing_elements),
(3) ADIBCO fields (may be incomplete),

Return ONLY a strict JSON object matching the provided schema.

STATEMENT TYPES (use ONLY these):
- rule: strong deontic AND explicit enforcement/sanction ("or else").
- norm: prescriptive expectation with a deontic but WITHOUT explicit sanction.
- strategy: action/plan/approach with actor + aim, no deontic requirement.
- principle: value/commitment/ethical high-level commitment (fairness, transparency, inclusion, etc.).
- other: not meaningfully one of the above.

SENIOR CODER RULES:
1) chunk_text is the source of truth. Do NOT override it with extracted fields.
2) Do NOT force-fit. "other" is acceptable if the chunk is informational or mixed.
3) Because these were Stage 1 "other":
   - If you upgrade to non-other, recoverable_from_other MUST be "Y".
   - If you keep "other", recoverable_from_other MUST be "N".
4) If final_statement_type=="other", you MUST set other_reason_bucket to a non-empty value from the allowed set.
   If final_statement_type!="other", other_reason_bucket MUST be "".
5) Provide evidence_span: a verbatim substring from chunk_text that justifies your decision (no paraphrase).
6) Provide institutional_signal (high/medium/low) reflecting how much institutional content is present.
7) Provide institutional_diagnosis (closed set) to explain WHY it is other or WHY it is recoverable.
8) missing_elements should list what prevents classification as rule/norm/strategy/principle (short phrases).
9) notes_for_humans: one short paragraph on what a human coder should notice.
10) recommended_pipeline_change: one short, concrete suggestion (e.g., better chunking, drop nav noise, detect glossary blocks).

OUTPUT:
Return ONLY valid JSON matching the schema. No extra text.
