#!/usr/bin/env python3
"""
STEP 8.3 — ADIBCO EXTRACTION (SENTENCE-LEVEL)

Implements the frozen handover spec exactly.

Inputs:
  - Chunked policy text (parquet): data/derived/step8_igt_full/chunks_full.parquet
    Expected columns (best effort):
      doc_id, chunk_id, chunk_text
    If chunk_text not present, will look for: text, chunk, chunk_normalized, content

Outputs (one row per institutional sentence):
  - data/derived/step8_igt_full/igt_statements_full.parquet
  - data/derived/step8_igt_full/igt_statements_full.csv
  - data/derived/step8_igt_full/_runtime_params.json (parser + lexicon hash)

Locked rules:
  - D: dependency-based (modal aux + deontic predicates), negation handling; lexical fallback allowed for modals.
  - I: dependency-based governed verb(s), allow conjoined aims; tight verb-phrase.
  - C: dependency harvesting (advcl + prep/obl phrases + legal hooks + exceptions) as lists.
  - A: grammatical only (nsubj of aim head; passive nsubjpass or agent), allow conjuncts, no bins.
  - B: lexicon-based search in aim span then full sentence.
  - O: local only (same sentence + next 1–2), cue-based.
  - Typing at 8.3 is provisional candidates only: strategy_candidate, norm_candidate, rule_candidate, other_low_confidence.
  - "Other" is typing ambiguity, NOT parse failure. Parse failures logged separately.

No stakeholder bins. No exhaustive lexical lists for A. No umbrella sanctions (these dealt with in 8.3B).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data/derived/step8_igt_full/chunks_full.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/derived/step8_igt_full"
DEFAULT_OUT_PARQUET = DEFAULT_OUT_DIR / "igt_statements_full.parquet"
DEFAULT_OUT_CSV = DEFAULT_OUT_DIR / "igt_statements_full.csv"
DEFAULT_RUNTIME = DEFAULT_OUT_DIR / "_runtime_params.json"

LEX_DIR = PROJECT_ROOT / "resources/lexicons"
B_LEX_PATH = LEX_DIR / "biophysical_object.txt"


# -----------------------------
# Utilities
# -----------------------------
def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def update_runtime_params(runtime_path: Path, updates: Dict) -> None:
    if runtime_path.exists():
        try:
            data = json.loads(runtime_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data.update(updates)
    runtime_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fmt_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def load_lexicon_terms(path: Path) -> List[str]:
    terms: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.append(s)
    return terms


def compile_lexicon_regex(terms: List[str]) -> re.Pattern:
    """
    Conservative common-noun lexicon match:
      - flexible whitespace in multiword terms
      - word boundaries
      - case-insensitive
    """
    if not terms:
        return re.compile(r"a\A")

    pats = []
    for t in terms:
        esc = re.escape(t.strip()).replace(r"\ ", r"\s+")
        pats.append(rf"\b{esc}\b")
    return re.compile(r"(?:%s)" % "|".join(pats), re.IGNORECASE)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# -----------------------------
# Locked cue sets
# -----------------------------
DEONTIC_MODALS = {"shall", "must", "may", "should", "can"}  # locked
DEONTIC_PREDICATES = {"require", "prohibit", "permit", "allow"}  # locked

# D class mapping
MODAL_CLASS = {
    "shall": "obligation",
    "must": "obligation",
    "should": "obligation",   # still prescriptive; we are NOT doing interpretive norm strength here
    "may": "permission",
    "can": "permission",
}
PRED_CLASS = {
    "require": "obligation",
    "permit": "permission",
    "allow": "permission",
    "prohibit": "prohibition",
}

COND_MARKERS = {"if", "when", "unless", "before", "after", "while", "provided"}  # locked
EXCEPT_MARKERS = {"except", "unless", "notwithstanding"}  # locked
LEGAL_HOOK_RE = re.compile(r"\b(subject\s+to|in\s+accordance\s+with|pursuant\s+to)\b", re.IGNORECASE)

# Strategy cues (for candidate typing ONLY)
STRATEGY_CUE_RE = re.compile(
    r"\b(strategy|roadmap|action\s+plan|programme|program|initiative|objective|priority|milestone)\b"
    r"|\b(aim(s)?\s+to|seek(s)?\s+to|intend(s)?\s+to|plan(s)?\s+to|commit(s)?\s+to|will\s+\w+)\b",
    re.IGNORECASE,
)

# Any-entity cue
ANY_ENTITY_RE = re.compile(r"\b(any\s+(person|entity|organisation|organization|legal\s+person|natural\s+person))\b", re.IGNORECASE)

# O local cues (locked family)
O_CUE_RE = re.compile(
    r"\b(failure\s+to\s+comply|non[-\s]?compliance|liable|fine|penalt(y|ies)|sanction(s)?|"
    r"offence|offense|revocation|suspension|result(s)?\s+in|lead(s)?\s+to)\b",
    re.IGNORECASE,
)

# -----------------------------
# Dependency helpers (spaCy)
# -----------------------------
def sent_text_span(sent) -> Tuple[int, int]:
    return (sent.start_char, sent.end_char)


def token_has_neg(tok) -> bool:
    return any(c.dep_ == "neg" for c in tok.children)


def subtree_text(tok) -> str:
    return normalize_ws(" ".join(t.text for t in tok.subtree))


def find_deontic(sent) -> Dict:
    """
    D — dependency-based only (modal aux + deontic predicates), with lexical fallback allowed.
    Returns a dict of D fields.
    """
    out = {
        "d_found": False,
        "d_lemma": None,
        "d_surface": None,
        "d_class": None,
        "d_polarity": None,
        "d_method": None,
        "d_head_i": None,   # token index (doc-level) of governed head
        "d_tok_i": None,    # token index (doc-level) of deontic token
    }

# 1) Modal auxiliaries via dependency
    for tok in sent:
        if tok.dep_ == "aux" and tok.lemma_.lower() in DEONTIC_MODALS:
            head = tok.head
            lemma = tok.lemma_.lower()
            d_class = MODAL_CLASS.get(lemma, "obligation")
            neg = token_has_neg(head) or token_has_neg(tok)
            polarity = "negative" if neg else "positive"
            # obligation + neg => prohibition; permission + neg => prohibition
            if neg and d_class in {"obligation", "permission"}:
                d_class = "prohibition"

            out.update(
                {
                    "d_found": True,
                    "d_lemma": lemma,
                    "d_surface": tok.text,
                    "d_class": d_class,
                    "d_polarity": polarity,
                    "d_method": "modal_aux",
                    "d_head_i": int(head.i),
                    "d_tok_i": int(tok.i),
                }
            )
            return out

    # 2) Deontic predicates via dependency
    for tok in sent:
        lem = tok.lemma_.lower()
        if lem in DEONTIC_PREDICATES and tok.pos_ in {"VERB", "ADJ"}:
            # Keep conservative: require/prohibit/permit/allow should behave like a predicate
            # We accept ROOT or attribute/acomp-ish predicate structures.
            if tok.dep_ in {"ROOT", "attr", "acomp"} or tok.dep_.startswith("advcl") or tok.dep_.startswith("ccomp"):
                d_class = PRED_CLASS[lem]
                neg = token_has_neg(tok)
                polarity = "negative" if neg else "positive"
                if neg and d_class in {"obligation", "permission"}:
                    d_class = "prohibition"
                out.update(
                    {
                        "d_found": True,
                        "d_lemma": lem,
                        "d_surface": tok.text,
                        "d_class": d_class,
                        "d_polarity": polarity,
                        "d_method": "deontic_predicate",
                        "d_head_i": int(tok.i),
                        "d_tok_i": int(tok.i),
                    }
                )
                return out
# 3) Lexical fallback (allowed by spec) — single-token scan only
    for tok in sent:
        if tok.lemma_.lower() in DEONTIC_MODALS:
            lemma = tok.lemma_.lower()
            d_class = MODAL_CLASS.get(lemma, "obligation")
            neg = token_has_neg(tok.head) or token_has_neg(tok)
            polarity = "negative" if neg else "positive"
            if neg and d_class in {"obligation", "permission"}:
                d_class = "prohibition"
            out.update(
                {
                    "d_found": True,
                    "d_lemma": lemma,
                    "d_surface": tok.text,
                    "d_class": d_class,
                    "d_polarity": polarity,
                    "d_method": "lexical_fallback",
                    "d_head_i": int(tok.head.i),
                    "d_tok_i": int(tok.i),
                }
            )
            return out

    return out

def find_aim(sent, d: Dict) -> Dict:
    """
    I — dependency-based.
    Modal case: governed ROOT/head verb.
    Predicate case: xcomp under required/prohibited/etc.
    Allow multiple conjoined aims.
    """
    out = {
        "i_head_lemma": None,     # joined if multiple
        "i_phrase_text": None,    # joined if multiple
        "i_has_conj": False,
        "i_method": None,
        "i_count": 0,
        "i_head_lemmas": [],      # list
        "i_phrase_texts": [],     # list
        "i_head_token_is": [],    # list of doc-level token indices
    }

    if not d.get("d_found"):
        # Strategy candidate still needs an Aim: use ROOT verb (dependency-based)
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is None:
            return out
        aims = [root] + [c for c in root.conjuncts if c.pos_ == "VERB"]
        out["i_method"] = "strategy_root"
    else:
        if d["d_method"] == "modal_aux":
            # deontic token's head is governed verb
            d_tok = next((t for t in sent if int(t.i) == int(d["d_tok_i"])), None)
            if d_tok is None:
                return out
            head = d_tok.head
            aims = [head] + [c for c in head.conjuncts if c.pos_ == "VERB"]
            out["i_method"] = "modal_head"
        else:
            # deontic predicate: aim is xcomp under predicate
            pred = next((t for t in sent if int(t.i) == int(d["d_tok_i"])), None)
            if pred is None:
                return out
            xcomps = [c for c in pred.children if c.dep_ in {"xcomp", "ccomp"} and c.pos_ == "VERB"]
            aims = []
            for x in xcomps:
                aims.append(x)
                aims.extend([c for c in x.conjuncts if c.pos_ == "VERB"])
            out["i_method"] = "predicate_xcomp"

    # Tight verb phrase expansion: verb + particle + direct object (if present)
    for a in aims:
        phrase_parts = [a.text]

        prt = next((c for c in a.children if c.dep_ == "prt"), None)
        if prt is not None:
            phrase_parts.append(prt.text)

        obj = next((c for c in a.children if c.dep_ in {"dobj", "obj"}), None)
        if obj is not None:
            phrase_parts.append(subtree_text(obj))

        phrase = normalize_ws(" ".join(phrase_parts))
        out["i_head_lemmas"].append(a.lemma_.lower())
        out["i_phrase_texts"].append(phrase)
        out["i_head_token_is"].append(int(a.i))

    out["i_count"] = len(out["i_head_lemmas"])
    out["i_has_conj"] = out["i_count"] > 1
    if out["i_count"] > 0:
        out["i_head_lemma"] = "|".join(out["i_head_lemmas"])
        out["i_phrase_text"] = "|".join(out["i_phrase_texts"])

    return out

def find_conditions(sent, aim_token_i: Optional[int]) -> Dict:
    """
    C — dependency harvesting:
      - advcl with markers (if/when/unless/before/after/while/provided)
      - legal hooks (subject to / in accordance with / pursuant to)
      - exceptions (except/unless/notwithstanding)
      - oblique/prep phrases (time/scope hooks) conservatively captured via subtree spans
    """
    out = {"c_texts": [], "c_types": [], "c_cues": [], "c_count": 0}

    # choose anchor: aim head if provided, else ROOT
    anchor = None
    if aim_token_i is not None:
        anchor = next((t for t in sent if int(t.i) == int(aim_token_i)), None)
    if anchor is None:
        anchor = next((t for t in sent if t.dep_ == "ROOT"), None)

    if anchor is None:
        return out

    # 1) advcl triggers
    for tok in sent:
        if tok.dep_ == "advcl":
            # marker word
            mark = next((c for c in tok.children if c.dep_ == "mark"), None)
            if mark is not None and mark.text.lower() in COND_MARKERS:
                out["c_texts"].append(subtree_text(tok))
                out["c_types"].append("trigger_clause")
                out["c_cues"].append(mark.text.lower())

    # 2) legal hooks (lexical phrase but harvested as condition text)
    m = LEGAL_HOOK_RE.search(sent.text)
    if m:
        # capture the clause/phrase from hook to end of sentence (tight enough for audit; later refinement ok)
        hook_start = m.start()
        hook_text = normalize_ws(sent.text[hook_start:])
        out["c_texts"].append(hook_text)
        out["c_types"].append("legal_hook")
        out["c_cues"].append(m.group(0).lower())

    # 3) exceptions markers
    for tok in sent:
        if tok.text.lower() in EXCEPT_MARKERS:
            # best effort: capture from exception word to end
            idx = tok.idx - sent.start_char
            ex_text = normalize_ws(sent.text[idx:])
            out["c_texts"].append(ex_text)
            out["c_types"].append("exception")
            out["c_cues"].append(tok.text.lower())

    # 4) oblique / prep phrases attached to anchor (conservative)
    # spaCy often uses 'prep' for PP head; we keep only those that look like constraints.
    for child in anchor.children:
        if child.dep_ == "prep":
            cue = child.text.lower()
            # keep time/scope-ish preps conservatively
            if cue in {"before", "after", "during", "within", "under", "upon", "for", "with", "without", "in"}:
                out["c_texts"].append(subtree_text(child))
                out["c_types"].append("prep_phrase")
                out["c_cues"].append(cue)

    out["c_count"] = len(out["c_texts"])
    return out
def find_attributes(sent, primary_aim_token_i: Optional[int]) -> Dict:
    """
    A — grammatical only:
      - nsubj of aim head (preferred)
      - passive: nsubjpass, or agent "by X" (obl/agent-ish)
      - allow conjuncts (stored via a_is_conjoined + raw span text)
      - special cases: expletive it -> implicit; no subject -> implicit; any person/entity -> any_entity
    """
    out = {
        "a_raw_text": None,
        "a_head": None,
        "a_class": None,   # explicit / any_entity / implicit
        "a_method": None,
        "a_is_conjoined": False,
    }

    anchor = None
    if primary_aim_token_i is not None:
        anchor = next((t for t in sent if int(t.i) == int(primary_aim_token_i)), None)
    if anchor is None:
        anchor = next((t for t in sent if t.dep_ == "ROOT"), None)

    if anchor is None:
        out["a_class"] = "implicit"
        out["a_method"] = "no_root"
        return out

    # subject of aim head
    subj = next((c for c in anchor.children if c.dep_ == "nsubj"), None)
    if subj is not None:
        raw = subtree_text(subj)
        head = subj.lemma_.lower()
        out.update({"a_raw_text": raw, "a_head": head, "a_method": "nsubj"})
    else:
        # passive subject
        psubj = next((c for c in anchor.children if c.dep_ == "nsubjpass"), None)
        if psubj is not None:
            raw = subtree_text(psubj)
            head = psubj.lemma_.lower()
            out.update({"a_raw_text": raw, "a_head": head, "a_method": "nsubjpass"})
        else:
            # agent phrase "by X"
            by_prep = next((c for c in anchor.children if c.dep_ == "prep" and c.text.lower() == "by"), None)
            if by_prep is not None:
                pobj = next((c for c in by_prep.children if c.dep_ == "pobj"), None)
                if pobj is not None:
                    raw = subtree_text(pobj)
                    head = pobj.lemma_.lower()
                    out.update({"a_raw_text": raw, "a_head": head, "a_method": "obl_agent_by"})

    # classify missingness / special cases
    if not out["a_raw_text"]:
        # expletive "it"
        it_subj = next((t for t in sent if t.dep_ == "expl" and t.text.lower() == "it"), None)
        if it_subj is not None:
            out["a_class"] = "implicit"
            out["a_method"] = "expletive_it"
            return out

        out["a_class"] = "implicit"
        if not out["a_method"]:
            out["a_method"] = "no_subject"
        return out

    # any_entity
    if ANY_ENTITY_RE.search(out["a_raw_text"] or ""):
        out["a_class"] = "any_entity"
    else:
        out["a_class"] = "explicit"

    # conjunction flag
    # if subject head has conjuncts, mark
    if subj is not None and list(subj.conjuncts):
        out["a_is_conjoined"] = True
    elif " and " in (out["a_raw_text"] or "").lower():
        out["a_is_conjoined"] = True

    return out

def find_b_object(sent_text: str, aim_phrase_text: Optional[str], b_rx: re.Pattern) -> Dict:
    """
    B — lexicon-based:
      - search aim span first (aim phrase text is small; fallback to full sentence)
    """
    out = {"b_found": False, "b_text": None, "b_cue": None}

    if aim_phrase_text:
        m = b_rx.search(aim_phrase_text)
        if m:
            out.update({"b_found": True, "b_text": m.group(0), "b_cue": "aim_span"})
            return out

    m2 = b_rx.search(sent_text)
    if m2:
        out.update({"b_found": True, "b_text": m2.group(0), "b_cue": "sentence"})
    return out


def find_o_local(sents: List, i_sent_idx: int) -> Dict:
    """
    O — local only in 8.3:
      - search same sentence, and next 1–2 sentences only
      - cue-based; extract minimal text (sentence containing cue)
    """
    out = {
        "o_local_present": False,
        "o_local_text": None,
        "o_local_type": None,
        "o_local_cue": None,
    }
    window_idxs = [i_sent_idx, i_sent_idx + 1, i_sent_idx + 2]
    window_idxs = [j for j in window_idxs if 0 <= j < len(sents)]

    for j in window_idxs:
        txt = sents[j].text
        m = O_CUE_RE.search(txt)
        if m:
            out["o_local_present"] = True
            out["o_local_text"] = normalize_ws(txt)
            out["o_local_type"] = "sanction_or_enforcement"
            out["o_local_cue"] = m.group(0).lower()
            return out

    return out

def provisional_type(d_found: bool, o_local_present: bool, sent_text: str, i_ok: bool) -> str:
    """
    Step 8.3 candidate typing only:
      - rule_candidate if D and local O
      - norm_candidate if D and no local O
      - strategy_candidate if no D but strategy cues present
      - other_low_confidence otherwise
    """
    if d_found:
        if not i_ok:
            return "other_low_confidence"
        return "rule_candidate" if o_local_present else "norm_candidate"

    # no D: strategy candidate only if strategy cues exist and aim exists
    if STRATEGY_CUE_RE.search(sent_text) and i_ok:
        return "strategy_candidate"

    return "other_low_confidence"

# -----------------------------
# Main
# -----------------------------
def resolve_text_column(df: pd.DataFrame) -> str:
    candidates = ["chunk_text", "text", "chunk", "chunk_normalized", "content"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find a chunk text column. Looked for: {candidates}. Found: {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / DEFAULT_OUT_PARQUET.name
    out_csv = out_dir / DEFAULT_OUT_CSV.name
    runtime_path = out_dir / DEFAULT_RUNTIME.name

    if not B_LEX_PATH.exists():
        raise FileNotFoundError(f"Missing required B lexicon at: {B_LEX_PATH}")

    b_terms = load_lexicon_terms(B_LEX_PATH)
    b_rx = compile_lexicon_regex(b_terms)

    nlp = spacy.load(args.spacy_model)
    # deterministic sentence splitter: use parser-produced sents; if absent, add sentencizer
    if not nlp.has_pipe("parser") and not nlp.has_pipe("senter"):
        nlp.add_pipe("sentencizer")

    update_runtime_params(
        runtime_path,
        {
            "step8_3_ran_at": dt.datetime.utcnow().isoformat() + "Z",
            "step8_3_input": str(in_path),
            "step8_3_output_parquet": str(out_parquet),
            "step8_3_output_csv": str(out_csv),
            "spacy_model": args.spacy_model,
            "spacy_version": spacy.__version__,
            "b_lexicon_path": str(B_LEX_PATH),
            "b_lexicon_sha256": sha256_file(B_LEX_PATH),
            "b_lexicon_terms_count": len(b_terms),
        },
    )
    df = pd.read_parquet(in_path)
    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    text_col = resolve_text_column(df)

    # identifiers best effort
    if "doc_id" not in df.columns:
        df["doc_id"] = df.get("document_id", None)
    if "chunk_id" not in df.columns:
        df["chunk_id"] = df.get("chunk_uid", df.index.astype(str))

    total = len(df)
    t0 = time.time()

    rows: List[Dict] = []
    parse_fail_rows: List[Dict] = []

    for idx, r in df.iterrows():
        doc_id = r.get("doc_id")
        chunk_id = r.get("chunk_id")
        chunk_text = str(r[text_col] or "")

        # Parse chunk
        try:
            doc = nlp(chunk_text)
            sents = list(doc.sents)
        except Exception as e:
            parse_fail_rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "error": str(e),
                    "text_len": len(chunk_text),
                }
            )
            continue

        for si, sent in enumerate(sents):
            sent_text = normalize_ws(sent.text)
            if not sent_text:
                continue

            # D
            d = find_deontic(sent)

            # I (if no D, still dependency-based root aim for strategy candidate evaluation)
            i = find_aim(sent, d)

            # decide if sentence is an "institutional sentence" row:
            # keep if D found OR strategy cues present (candidate), else skip entirely
            keep = bool(d["d_found"]) or bool(STRATEGY_CUE_RE.search(sent_text))
            if not keep:
                continue

            # primary aim head token index
            primary_aim_token_i = i["i_head_token_is"][0] if i["i_head_token_is"] else None

            # C
            c = find_conditions(sent, primary_aim_token_i)

            # A
            a = find_attributes(sent, primary_aim_token_i)

            # B (aim span = tight i_phrase_text; fallback sentence)
            b = find_b_object(sent_text, i.get("i_phrase_text"), b_rx)

            # O local (same sentence + next 1–2)
            o = find_o_local(sents, si)

            # provisional type candidate only
            i_ok = bool(i.get("i_head_lemma"))
            stmt_type = provisional_type(d["d_found"], o["o_local_present"], sent_text, i_ok)

            rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "sentence_text": sent_text,
                    "sentence_index_in_chunk": si,

                    # D fields
                    "d_lemma": d["d_lemma"],
                    "d_surface": d["d_surface"],
                    "d_class": d["d_class"],
                    "d_polarity": d["d_polarity"],
                    "d_method": d["d_method"],

                    # I fields
                    "i_head_lemma": i["i_head_lemma"],
                    "i_phrase_text": i["i_phrase_text"],
                    "i_has_conj": i["i_has_conj"],
                    "i_method": i["i_method"],

                    # C fields (stored as joined lists)
                    "c_texts": "|".join(c["c_texts"]) if c["c_texts"] else None,
                    "c_types": "|".join(c["c_types"]) if c["c_types"] else None,
                    "c_cues": "|".join(c["c_cues"]) if c["c_cues"] else None,
                    "c_count": int(c["c_count"]),

                    # B fields
                    "b_text": b["b_text"],
                    "b_cue": b["b_cue"],
                    "b_found": bool(b["b_found"]),

                    # A fields
                    "a_raw_text": a["a_raw_text"],
                    "a_head": a["a_head"],
                    "a_class": a["a_class"],
                    "a_method": a["a_method"],
                    "a_is_conjoined": bool(a["a_is_conjoined"]),

                    # O local fields (only)
                    "o_local_present": bool(o["o_local_present"]),
                    "o_local_text": o["o_local_text"],
                    "o_local_type": o["o_local_type"],
                    "o_local_cue": o["o_local_cue"],

                    # Provisional typing (candidate only)
                    "statement_type_candidate": stmt_type,
                }
            )
        if args.report_every and (len(rows) > 0) and (idx % args.report_every == 0):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(1e-6, elapsed)
            eta = (total - (idx + 1)) / max(1e-6, rate)
            print(f"[step8_3] processed {idx+1}/{total} chunks | rows={len(rows)} | elapsed={fmt_hhmmss(elapsed)} | eta={fmt_hhmmss(eta)}")

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_parquet, index=False)
    out_df.to_csv(out_csv, index=False)

    # write parse failures separately for audit (not "other")
    if parse_fail_rows:
        pd.DataFrame(parse_fail_rows).to_csv(out_dir / "igt_parse_failures.csv", index=False)

    elapsed = time.time() - t0
    update_runtime_params(
        runtime_path,
        {
            "step8_3_rows_written": int(len(out_df)),
            "step8_3_chunks_processed": int(total),
            "step8_3_elapsed_seconds": float(elapsed),
            "step8_3_parse_failures": int(len(parse_fail_rows)),
        },
    )

    print(f"[step8_3] Wrote: {out_parquet}")
    print(f"[step8_3] Wrote: {out_csv}")
    if parse_fail_rows:
        print(f"[step8_3] Wrote: {out_dir / 'igt_parse_failures.csv'}")
    print(f"[step8_3] Done in {fmt_hhmmss(elapsed)}")


if __name__ == "__main__":
    main()
