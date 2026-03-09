#!/usr/bin/env python3
"""
Step 8.5h — General-scope / umbrella applicability inquiry for the education corpus.

Purpose
-------
Provide a bounded robustness test for the concern that some governance rules apply
broadly to education through document- or section-level scope, even when the
immediate sentence does not name teachers, students, or schools as A actors.

This is NOT a replacement for the sentence-level actor analysis. It is a
conservative supplement that asks:
- which education-relevant rule statements sit inside broad / general-scope
  sections?
- which rule statements themselves contain broad applicability cues?
- how often do such rules still omit pedagogical actors in A?
- does adding a general-scope layer materially change the actor picture?

Default inputs
--------------
- data/derived/step1_texts/docs_normalized_text/
- data/derived/step3_chunks_spacy/chunks_spacy.jsonl
- data/derived/step6_chunks_edu/chunks_edu.jsonl
- data/derived/step8_igt_chunks_edu/igt_statements_full.parquet

Default outputs
---------------
- data/derived/step8_analysis/general_scope_umbrella_edu/
    - general_scope_sections_edu_docs.csv
    - general_scope_rules_edu.csv
    - general_scope_summary.csv
    - general_scope_doc_summary.csv
    - general_scope_actor_comparison.csv
    - general_scope_exemplars_direct_pedagogical_A.csv
    - general_scope_exemplars_inherited_scope_only.csv
    - general_scope_exemplars_sentence_scope_only.csv
    - general_scope_exemplars_section_scope_only.csv
    - general_scope_summary.md
    - run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    return re.sub(r"\s+", " ", textify(s)).strip()



def normalize_stmt_type(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()



def build_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)



def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0



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



def iter_doc_ids_from_chunks_jsonl(path: Path) -> set[str]:
    doc_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or obj.get("source_doc") or obj.get("document_id")
            if doc_id is not None:
                doc_ids.add(str(doc_id))
    return doc_ids



def load_chunk_spans(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            char_start = obj.get("char_start", None)
            char_end = obj.get("char_end", None)
            if char_start is None or char_end is None:
                continue
            try:
                cs = int(char_start)
                ce = int(char_end)
            except Exception:
                continue
            if cs < 0 or ce < 0:
                continue
            rows.append(
                {
                    "doc_id": str(obj.get("doc_id") or obj.get("source_doc") or obj.get("document_id") or ""),
                    "chunk_id": str(obj.get("chunk_id") or obj.get("chunk_uid") or obj.get("id") or i),
                    "chunk_char_start": cs,
                    "chunk_char_end": ce,
                }
            )
    return pd.DataFrame(rows)
# -----------------------------------------------------------------------------
# Actor groups (aligned to step8_5d)
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
        r"\bmember state(s)?\b",
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
PEDAGOGICAL_GROUPS = ["educators_teachers", "students_learners", "schools_institutions"]
DEFAULT_RULE_LABELS = {"rule_candidate", "rule", "rules"}

# -----------------------------------------------------------------------------
# General-scope / umbrella detection
# -----------------------------------------------------------------------------

HEADING_MAX_CHARS = 160

HEADING_SCOPE_CORE_RE = re.compile(
    r"\b(scope|application|applicability|general provisions?|definitions?)\b",
    re.IGNORECASE,
)

HEADING_GENERAL_RULE_RE = re.compile(
    r"\b(obligations?|duties|responsibilities|requirements?|prohibited practices?|"
    r"high[- ]risk|transparency|human oversight|risk management|data governance|"
    r"quality management|accuracy|robustness|monitoring|conformity|providers?|"
    r"deployers?|users?|authorities?|institutions?)\b",
    re.IGNORECASE,
)

HEADING_EDU_RE = re.compile(
    r"\b(education|educational|school|schools|university|universities|college|"
    r"student|students|learner|learners|teacher|teachers|vocational training|"
    r"admission|assessment|grading|proctoring)\b",
    re.IGNORECASE,
)

BODY_APPLICABILITY_RE = re.compile(
    r"\b(this (regulation|law|act|directive|framework|policy) applies to|"
    r"shall apply to|applies to|for the purposes of this|within the meaning of this|"
    r"in the context of|intended to be used to|used to determine|used to evaluate|"
    r"used for (admission|assessment|grading|proctoring)|in the education sector|"
    r"in education and vocational training|in educational institutions?)\b",
    re.IGNORECASE,
)

BODY_UNIVERSAL_ACTOR_RE = re.compile(
    r"\b(all|any|each|every|no)\s+"
    r"(provider|providers|deployer|deployers|user|users|member state|member states|"
    r"competent authority|competent authorities|educational institution|educational institutions|"
    r"school|schools|university|universities|institution|institutions|public authority|public authorities)\b",
    re.IGNORECASE,
)

BODY_GENERIC_RULE_ACTOR_RE = re.compile(
    r"\b(member states?|providers?|deployers?|users?|competent authorities?|"
    r"public authorities?|educational institutions?|schools?|universities?|institutions?)\b",
    re.IGNORECASE,
)

BODY_MODAL_RE = re.compile(r"\b(shall|must|may not|is prohibited|are prohibited|required to)\b", re.IGNORECASE)

SENTENCE_SCOPE_RE = re.compile(
    r"\b(this (regulation|law|act|directive|framework|policy) applies to|shall apply to|applies to|"
    r"for the purposes of this|within the meaning of this|all providers|all deployers|all users|"
    r"all member states|all educational institutions|any provider|any deployer|each provider|each deployer|"
    r"member states shall ensure|providers shall|deployers shall|users shall|educational institutions shall|"
    r"schools shall|universities shall|competent authorities shall)\b",
    re.IGNORECASE,
)



def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > HEADING_MAX_CHARS:
        return False
    if s.endswith(":"):
        return True
    if s.isupper() and len(s) >= 5:
        return True
    words = re.findall(r"[A-Za-z0-9]+", s)
    if 1 <= len(words) <= 14:
        caps = sum(1 for w in words if w and w[0].isupper())
        nums = sum(1 for w in words if w.isdigit() or re.fullmatch(r"[ivxlcdm]+", w.lower()))
        return (caps + nums) / max(len(words), 1) >= 0.6
    return False



def classify_section(heading: str, body_text: str) -> tuple[bool, str, bool]:
    heading_n = normalize_space(heading)
    body_n = normalize_space(body_text)

    heading_scope = bool(HEADING_SCOPE_CORE_RE.search(heading_n))
    heading_general_rule = bool(HEADING_GENERAL_RULE_RE.search(heading_n))
    heading_edu = bool(HEADING_EDU_RE.search(heading_n))
    body_applicability = bool(BODY_APPLICABILITY_RE.search(body_n))
    body_universal_actor = bool(BODY_UNIVERSAL_ACTOR_RE.search(body_n))
    body_generic_actor = bool(BODY_GENERIC_RULE_ACTOR_RE.search(body_n))
    body_modal = bool(BODY_MODAL_RE.search(body_n))
    body_edu = bool(HEADING_EDU_RE.search(body_n))

    section_general_scope = False
    reason = "none"

    if heading_scope:
        section_general_scope = True
        reason = "scope_heading"
    elif heading_general_rule and (body_applicability or body_universal_actor):
        section_general_scope = True
        reason = "general_rule_heading_plus_broad_body"
    elif body_applicability and (heading_edu or body_edu or body_generic_actor):
        section_general_scope = True
        reason = "applicability_body_cue"
    elif body_universal_actor and heading_general_rule and body_modal:
        section_general_scope = True
        reason = "universal_actor_plus_modal"

    edu_signal = heading_edu or body_edu
    return section_general_scope, reason, edu_signal



def extract_sections_from_doc_text(doc_id: str, text: str) -> pd.DataFrame:
    lines = text.splitlines(True)
    offsets: List[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    headings: List[Tuple[int, str]] = []
    for i, ln in enumerate(lines):
        raw = ln.strip()
        if looks_like_heading(raw):
            headings.append((i, raw.rstrip(":").strip()))

    cols = [
        "doc_id",
        "section_id",
        "heading_text",
        "section_char_start",
        "section_char_end",
        "section_text",
        "general_scope_flag",
        "general_scope_reason",
        "edu_signal_in_section",
    ]

    if not headings:
        return pd.DataFrame(columns=cols)

    rows: List[dict] = []
    for h_idx, (line_i, heading) in enumerate(headings):
        start_line = line_i
        end_line = headings[h_idx + 1][0] if (h_idx + 1) < len(headings) else len(lines)
        section_char_start = offsets[start_line]
        section_char_end = offsets[end_line - 1] + len(lines[end_line - 1])
        section_text = "".join(lines[start_line:end_line]).strip("\n")
        body_only = normalize_space(" ".join(ln.strip() for ln in lines[start_line + 1:end_line]))
        is_general_scope, reason, edu_signal = classify_section(heading, body_only)
        rows.append(
            {
                "doc_id": doc_id,
                "section_id": f"{doc_id}::GS{h_idx:04d}",
                "heading_text": heading,
                "section_char_start": section_char_start,
                "section_char_end": section_char_end,
                "section_text": section_text,
                "general_scope_flag": is_general_scope,
                "general_scope_reason": reason,
                "edu_signal_in_section": edu_signal,
            }
        )

    return pd.DataFrame(rows, columns=cols)



def extract_general_scope_sections(docs_dir: Path, edu_doc_ids: set[str], report_every: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    files = sorted(docs_dir.glob("*.txt"))
    t0 = time.time()
    seen = 0
    for fp in files:
        doc_id = fp.stem
        if edu_doc_ids and doc_id not in edu_doc_ids:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sec = extract_sections_from_doc_text(doc_id, text)
        if not sec.empty:
            sec = sec[sec["general_scope_flag"] == True].copy()
            if not sec.empty:
                rows.append(sec)
        seen += 1
        if report_every and seen % report_every == 0:
            print(f"[extract] edu_docs={seen:,} sections={sum(len(x) for x in rows):,} elapsed_min={(time.time()-t0)/60:.1f}")

    if not rows:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "section_id",
                "heading_text",
                "section_char_start",
                "section_char_end",
                "section_text",
                "general_scope_flag",
                "general_scope_reason",
                "edu_signal_in_section",
            ]
        )
    return pd.concat(rows, ignore_index=True)



def interval_overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (b0 < a1)



def attach_section_flags_to_statements(
    statements: pd.DataFrame,
    chunk_spans: pd.DataFrame,
    sections: pd.DataFrame,
    report_every: int,
) -> pd.DataFrame:
    st = statements.copy()
    st["doc_id"] = st["doc_id"].astype(str)
    st["chunk_id"] = st["chunk_id"].astype(str)

    spans = chunk_spans.copy()
    spans["doc_id"] = spans["doc_id"].astype(str)
    spans["chunk_id"] = spans["chunk_id"].astype(str)

    st = st.merge(spans, on=["doc_id", "chunk_id"], how="left")

    sec_by_doc: Dict[str, List[Tuple[int, int, str, str, str, bool]]] = {}
    if not sections.empty:
        for doc_id, g in sections.groupby("doc_id"):
            sec_by_doc[str(doc_id)] = [
                (
                    int(r["section_char_start"]),
                    int(r["section_char_end"]),
                    str(r["section_id"]),
                    str(r["heading_text"]),
                    str(r["general_scope_reason"]),
                    bool(r.get("edu_signal_in_section", False)),
                )
                for _, r in g.iterrows()
            ]

    sec_flags: List[bool] = []
    sec_ids: List[str] = []
    sec_headings: List[str] = []
    sec_reasons: List[str] = []
    sec_edu_flags: List[bool] = []

    t0 = time.time()
    for i, r in enumerate(st.itertuples(index=False), start=1):
        doc_id = textify(getattr(r, "doc_id", ""))
        cs = getattr(r, "chunk_char_start", None)
        ce = getattr(r, "chunk_char_end", None)
        matched = False
        matched_id = ""
        matched_heading = ""
        matched_reason = ""
        matched_edu = False

        if doc_id and pd.notna(cs) and pd.notna(ce) and doc_id in sec_by_doc:
            cs_i = int(cs)
            ce_i = int(ce)
            for s0, s1, sid, shead, sreason, sedu in sec_by_doc[doc_id]:
                if interval_overlaps(cs_i, ce_i, s0, s1):
                    matched = True
                    matched_id = sid
                    matched_heading = shead
                    matched_reason = sreason
                    matched_edu = bool(sedu)
                    break

        sec_flags.append(matched)
        sec_ids.append(matched_id)
        sec_headings.append(matched_heading)
        sec_reasons.append(matched_reason)
        sec_edu_flags.append(matched_edu)

        if report_every and i % report_every == 0:
            print(f"[link] statements={i:,}/{len(st):,} elapsed_min={(time.time()-t0)/60:.1f}")

    st["section_general_scope_flag"] = sec_flags
    st["section_general_scope_section_id"] = sec_ids
    st["section_general_scope_heading"] = sec_headings
    st["section_general_scope_reason"] = sec_reasons
    st["section_edu_signal"] = sec_edu_flags
    return st


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="data/derived/step1_texts/docs_normalized_text")
    ap.add_argument("--chunks-spacy-jsonl", default="data/derived/step3_chunks_spacy/chunks_spacy.jsonl")
    ap.add_argument("--edu-chunks-jsonl", default="data/derived/step6_chunks_edu/chunks_edu.jsonl")
    ap.add_argument("--igt-parquet", default="data/derived/step8_igt_chunks_edu/igt_statements_full.parquet")
    ap.add_argument("--out-dir", default="data/derived/step8_analysis/general_scope_umbrella_edu")
    ap.add_argument("--rule-labels", nargs="*", default=sorted(DEFAULT_RULE_LABELS))
    ap.add_argument("--report-every", type=int, default=10000)
    ap.add_argument("--top-n", type=int, default=200)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    actor_patterns = compile_group_patterns(ACTOR_GROUP_PATTERNS)
    rule_labels = {str(x).strip().lower() for x in args.rule_labels}

    edu_doc_ids = iter_doc_ids_from_chunks_jsonl(Path(args.edu_chunks_jsonl))
    sections = extract_general_scope_sections(Path(args.docs_dir), edu_doc_ids, report_every=max(1, args.report_every // 10))
    if sections.empty:
        print("[warn] no general-scope sections detected in education docs")
    safe_to_csv(sections, out_dir / "general_scope_sections_edu_docs.csv")

    igt = maybe_read_table(Path(args.igt_parquet)).copy()
    required = {"doc_id", "chunk_id", "sentence_text", "statement_type_candidate", "a_raw_text", "a_class"}
    missing = sorted(list(required - set(igt.columns)))
    if missing:
        raise ValueError(f"Missing expected IGT columns: {missing}")

    igt["_stmt_type_norm"] = normalize_stmt_type(igt["statement_type_candidate"])
    igt["_is_rule"] = igt["_stmt_type_norm"].isin(rule_labels)
    rules = igt.loc[igt["_is_rule"]].copy()

    chunk_spans = load_chunk_spans(Path(args.chunks_spacy_jsonl))
    if chunk_spans.empty:
        raise ValueError("No chunk spans loaded from chunks_spacy.jsonl; cannot link section scope to rule statements")

    rules = attach_section_flags_to_statements(rules, chunk_spans, sections, report_every=args.report_every)

    rules["sentence_general_scope_flag"] = rules["sentence_text"].fillna("").astype(str).str.contains(SENTENCE_SCOPE_RE)
    rules["any_general_scope_flag"] = rules["sentence_general_scope_flag"] | rules["section_general_scope_flag"]
    rules["general_scope_source"] = "none"
    rules.loc[rules["sentence_general_scope_flag"] & ~rules["section_general_scope_flag"], "general_scope_source"] = "sentence_only"
    rules.loc[~rules["sentence_general_scope_flag"] & rules["section_general_scope_flag"], "general_scope_source"] = "section_only"
    rules.loc[rules["sentence_general_scope_flag"] & rules["section_general_scope_flag"], "general_scope_source"] = "sentence_and_section"

    rules["a_group_hits"] = rules["a_raw_text"].fillna("").astype(str).apply(lambda s: "|".join(match_groups(s, actor_patterns, ACTOR_GROUP_ORDER)))
    rules["sentence_pedagogical_hits"] = rules["sentence_text"].fillna("").astype(str).apply(
        lambda s: "|".join(match_groups(s, actor_patterns, PEDAGOGICAL_GROUPS))
    )
    rules["a_pedagogical_hits"] = rules["a_raw_text"].fillna("").astype(str).apply(
        lambda s: "|".join(match_groups(s, actor_patterns, PEDAGOGICAL_GROUPS))
    )
    rules["explicit_A_flag"] = rules["a_class"].fillna("").astype(str).str.strip().str.lower().eq("explicit")
    rules["direct_pedagogical_A_explicit"] = rules["explicit_A_flag"] & rules["a_pedagogical_hits"].ne("")
    rules["inherited_scope_only"] = rules["any_general_scope_flag"] & (~rules["direct_pedagogical_A_explicit"])

    keep_cols = [
        c
        for c in [
            "doc_id",
            "chunk_id",
            "sentence_index_in_chunk",
            "statement_type_candidate",
            "a_class",
            "a_raw_text",
            "a_group_hits",
            "a_pedagogical_hits",
            "sentence_pedagogical_hits",
            "sentence_general_scope_flag",
            "section_general_scope_flag",
            "general_scope_source",
            "section_general_scope_section_id",
            "section_general_scope_heading",
            "section_general_scope_reason",
            "section_edu_signal",
            "direct_pedagogical_A_explicit",
            "inherited_scope_only",
            "sentence_text",
        ]
        if c in rules.columns
    ]
    safe_to_csv(rules[keep_cols].copy(), out_dir / "general_scope_rules_edu.csv")

    # Summary
    n_rules = int(len(rules))
    n_any_scope = int(rules["any_general_scope_flag"].sum())
    n_sentence_scope = int(rules["sentence_general_scope_flag"].sum())
    n_section_scope = int(rules["section_general_scope_flag"].sum())
    n_both = int((rules["sentence_general_scope_flag"] & rules["section_general_scope_flag"]).sum())
    n_direct_ped = int(rules["direct_pedagogical_A_explicit"].sum())
    n_scope_direct_ped = int((rules["any_general_scope_flag"] & rules["direct_pedagogical_A_explicit"]).sum())
    n_scope_inherited = int(rules["inherited_scope_only"].sum())

    summary = pd.DataFrame(
        [
            {
                "total_edu_rules": n_rules,
                "rules_with_any_general_scope": n_any_scope,
                "pct_rules_with_any_general_scope": pct(n_any_scope, n_rules),
                "rules_with_sentence_scope_cue": n_sentence_scope,
                "rules_with_section_scope_link": n_section_scope,
                "rules_with_both_sentence_and_section_scope": n_both,
                "rules_with_direct_pedagogical_A_explicit": n_direct_ped,
                "general_scope_rules_with_direct_pedagogical_A_explicit": n_scope_direct_ped,
                "general_scope_rules_inherited_scope_only": n_scope_inherited,
                "pct_general_scope_rules_inherited_scope_only_of_all_rules": pct(n_scope_inherited, n_rules),
                "pct_general_scope_rules_inherited_scope_only_within_general_scope_rules": pct(n_scope_inherited, n_any_scope),
            }
        ]
    )
    safe_to_csv(summary, out_dir / "general_scope_summary.csv")

    # Doc summary
    doc_summary = (
        rules.groupby("doc_id")
        .agg(
            total_rules=("doc_id", "size"),
            rules_with_any_general_scope=("any_general_scope_flag", "sum"),
            rules_with_sentence_scope=("sentence_general_scope_flag", "sum"),
            rules_with_section_scope=("section_general_scope_flag", "sum"),
            direct_pedagogical_A_explicit=("direct_pedagogical_A_explicit", "sum"),
            inherited_scope_only=("inherited_scope_only", "sum"),
        )
        .reset_index()
    )
    doc_summary["pct_rules_with_any_general_scope"] = (100.0 * doc_summary["rules_with_any_general_scope"] / doc_summary["total_rules"].clip(lower=1)).round(2)
    doc_summary = doc_summary.sort_values(["rules_with_any_general_scope", "total_rules"], ascending=[False, False])
    safe_to_csv(doc_summary, out_dir / "general_scope_doc_summary.csv")

    # Actor comparison among explicit-A rules
    explicit_rules = rules.loc[rules["explicit_A_flag"]].copy()
    explicit_scope = explicit_rules.loc[explicit_rules["any_general_scope_flag"]].copy()
    actor_rows: List[dict] = []
    for group in ACTOR_GROUP_ORDER:
        all_ct = int(explicit_rules["a_group_hits"].fillna("").astype(str).str.contains(fr"(^|\|){re.escape(group)}(\||$)").sum())
        scope_ct = int(explicit_scope["a_group_hits"].fillna("").astype(str).str.contains(fr"(^|\|){re.escape(group)}(\||$)").sum())
        actor_rows.append(
            {
                "actor_group": group,
                "explicit_A_all_rules_count": all_ct,
                "explicit_A_all_rules_pct": pct(all_ct, len(explicit_rules)),
                "explicit_A_general_scope_rules_count": scope_ct,
                "explicit_A_general_scope_rules_pct": pct(scope_ct, len(explicit_scope)),
            }
        )
    actor_comp = pd.DataFrame(actor_rows).sort_values("explicit_A_general_scope_rules_count", ascending=False)
    safe_to_csv(actor_comp, out_dir / "general_scope_actor_comparison.csv")

    # Exemplars
    direct = rules.loc[rules["any_general_scope_flag"] & rules["direct_pedagogical_A_explicit"]].copy()
    inherited = rules.loc[rules["inherited_scope_only"]].copy()
    sent_only = rules.loc[rules["general_scope_source"].eq("sentence_only")].copy()
    sec_only = rules.loc[rules["general_scope_source"].eq("section_only")].copy()
    sort_cols = [c for c in ["doc_id", "chunk_id", "sentence_index_in_chunk"] if c in rules.columns]
    if sort_cols:
        direct = direct.sort_values(sort_cols)
        inherited = inherited.sort_values(sort_cols)
        sent_only = sent_only.sort_values(sort_cols)
        sec_only = sec_only.sort_values(sort_cols)

    safe_to_csv(direct.head(args.top_n)[keep_cols], out_dir / "general_scope_exemplars_direct_pedagogical_A.csv")
    safe_to_csv(inherited.head(args.top_n)[keep_cols], out_dir / "general_scope_exemplars_inherited_scope_only.csv")
    safe_to_csv(sent_only.head(args.top_n)[keep_cols], out_dir / "general_scope_exemplars_sentence_scope_only.csv")
    safe_to_csv(sec_only.head(args.top_n)[keep_cols], out_dir / "general_scope_exemplars_section_scope_only.csv")

    top_docs = doc_summary.head(15).copy()
    md_lines = [
        "# General-scope / umbrella applicability inquiry (education corpus)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a bounded robustness layer. It checks whether education-relevant rule statements may inherit broad applicability from sentence cues or document/section scope, even when pedagogical actors are not named directly in A.",
        "",
        "## Summary",
        build_markdown_table(summary),
        "",
        "## Top documents by general-scope rule count",
        build_markdown_table(top_docs) if not top_docs.empty else "No document summary rows.",
        "",
        "## Actor comparison (explicit-A rules)",
        build_markdown_table(actor_comp.head(15)) if not actor_comp.empty else "No actor comparison rows.",
        "",
        "Interpretation target: compare general_scope_rules_inherited_scope_only against general_scope_rules_with_direct_pedagogical_A_explicit. If inherited-scope rules are common but still dominated by non-pedagogical actors in A, then umbrella applicability qualifies the finding without overturning it.",
    ]
    (out_dir / "general_scope_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "docs_dir": args.docs_dir,
            "chunks_spacy_jsonl": args.chunks_spacy_jsonl,
            "edu_chunks_jsonl": args.edu_chunks_jsonl,
            "igt_parquet": args.igt_parquet,
        },
        "rule_labels": sorted(rule_labels),
        "counts": {
            "edu_docs": len(edu_doc_ids),
            "general_scope_sections": int(len(sections)),
            "total_edu_rules": n_rules,
            "rules_with_any_general_scope": n_any_scope,
            "direct_pedagogical_A_explicit": n_direct_ped,
            "general_scope_rules_inherited_scope_only": n_scope_inherited,
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
