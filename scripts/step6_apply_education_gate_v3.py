#!/usr/bin/env python3
"""
Education Gate v3: source-backed, tiered education-sector relevance gate.

Purpose
-------
Identify paragraph-based chunks with explicit education-sector evidence using
an externally sourced vocabulary derived from ERIC, UNESCO Thesaurus, and
ISCED.

Gate logic
----------
1. Anchor terms can independently establish education-sector relevance.
2. Contextual terms are recorded but cannot independently establish relevance.
3. Tag-only terms are recorded for downstream characterization and never affect
   inclusion.

The script preserves the complete source chunk once an anchor establishes the
education context. This allows downstream institutional-statement extraction to
use the full chunk context rather than requiring each individual sentence to
repeat education terminology.

Inputs
------
data/derived/step4_chunks_tagged/chunks_normalized_tagged.jsonl
resources/edu_lexicon_v3.yml

Outputs
-------
data/derived/step6_education_gate_v3/education_chunks_v3.jsonl
data/derived/step6_education_gate_v3/education_gate_v3_decisions.csv
data/derived/step6_education_gate_v3/education_gate_v3_positive_review.csv
data/derived/step6_education_gate_v3/education_gate_v3_summary.csv
data/derived/step6_education_gate_v3/education_gate_v3_manifest.json
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm


WS_RE = re.compile(r"\s+")


def normalize(text):
    return WS_RE.sub(" ", str(text).lower()).strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_text(record):
    for key in (
        "text_normalized",
        "text_norm",
        "text",
        "chunk_text",
        "chunk_text_normalized",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("No recognized text field found in chunk record.")


def extract_doc_id(record):
    for key in ("doc_id", "document_id", "docid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return "UNKNOWN_DOC"


def extract_chunk_id(record, line_number):
    for key in ("chunk_id", "id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"line_{line_number}"


def compile_term_records(grouped_items):
    """Compile YAML vocabulary records while preserving provenance metadata."""
    compiled = []

    for group, items in grouped_items.items():
        for item in items:
            canonical = str(item.get("canonical", "")).strip()
            source = str(item.get("source", "")).strip()
            source_status = str(item.get("source_status", "")).strip()
            source_url = str(item.get("source_url", "")).strip()

            for raw_term in item.get("terms", []):
                raw_term = str(raw_term).strip()
                if not raw_term:
                    continue

                term_norm = normalize(raw_term)
                pattern = re.compile(
                    rf"(?<!\w){re.escape(term_norm)}(?!\w)",
                    flags=re.IGNORECASE,
                )

                compiled.append({
                    "group": group,
                    "canonical": canonical,
                    "term": raw_term,
                    "pattern": pattern,
                    "source": source,
                    "source_status": source_status,
                    "source_url": source_url,
                })

    return compiled


def find_hits(text, compiled_records):
    text_norm = normalize(text)
    hits = []
    seen = set()

    for item in compiled_records:
        if item["pattern"].search(text_norm):
            key = (item["group"], item["canonical"], item["term"])
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "group": item["group"],
                "canonical": item["canonical"],
                "term": item["term"],
                "source": item["source"],
                "source_status": item["source_status"],
                "source_url": item["source_url"],
            })

    return hits


def unique_join(hits, key):
    values = []
    seen = set()
    for hit in hits:
        value = str(hit.get(key, "")).strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return "|".join(values)


def groups_set(hits):
    return {hit["group"] for hit in hits}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--infile",
        default=(
            "data/derived/step4_chunks_tagged/"
            "chunks_normalized_tagged.jsonl"
        ),
    )

    parser.add_argument(
        "--lexicon",
        default="resources/edu_lexicon_v3.yml",
    )

    parser.add_argument(
        "--outdir",
        default="data/derived/step6_education_gate_v3",
    )

    args = parser.parse_args()

    infile = Path(args.infile)
    lexicon_path = Path(args.lexicon)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lexicon = yaml.safe_load(lexicon_path.read_text(encoding="utf-8"))

    version = lexicon.get("version", "edu_gate_v3")
    status = lexicon.get("status", "unspecified")

    anchors = compile_term_records(lexicon.get("anchors", {}))
    contextual = compile_term_records(lexicon.get("contextual_terms", {}))
    tag_only = compile_term_records(lexicon.get("tag_only_terms", {}))

    chunks_out = outdir / "education_chunks_v3.jsonl"
    decisions_out = outdir / "education_gate_v3_decisions.csv"
    review_out = outdir / "education_gate_v3_positive_review.csv"
    summary_out = outdir / "education_gate_v3_summary.csv"
    manifest_out = outdir / "education_gate_v3_manifest.json"

    total = 0
    kept = 0
    docs_all = set()
    docs_kept = set()

    anchor_group_counts = Counter()
    anchor_canonical_counts = Counter()
    anchor_term_counts = Counter()
    contextual_group_counts = Counter()
    tag_group_counts = Counter()

    # Descriptive subgroup counts among INCLUDED chunks only.
    subgroup_counts = Counter()

    k12_groups = {
        "k12",
        "k12_actor",
        "k12_edtech",
        "k12_governance",
        "k12_pedagogy",
    }
    higher_ed_groups = {
        "higher_ed",
        "higher_ed_actor",
        "higher_ed_governance",
    }

    decision_fields = [
        "chunk_id",
        "doc_id",
        "edu_match",
        "edu_match_rule",
        "anchor_groups",
        "anchor_canonicals",
        "anchor_terms",
        "contextual_groups",
        "contextual_canonicals",
        "contextual_terms",
        "tag_only_groups",
        "tag_only_canonicals",
        "tag_only_terms",
        "k12_explicit_signal",
        "higher_ed_explicit_signal",
        "formal_vet_signal",
        "pedagogy_signal",
        "teacher_development_signal",
        "workforce_capacity_signal",
        "higher_ed_research_signal",
    ]

    review_fields = decision_fields + ["text"]

    total_input_chunks = sum(
        1 for line in infile.open("r", encoding="utf-8")
        if line.strip()
    )

    with (
        infile.open("r", encoding="utf-8") as source,
        chunks_out.open("w", encoding="utf-8") as chunks_fh,
        decisions_out.open("w", newline="", encoding="utf-8") as decisions_fh,
        review_out.open("w", newline="", encoding="utf-8") as review_fh,
    ):
        decision_writer = csv.DictWriter(
            decisions_fh,
            fieldnames=decision_fields,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )
        decision_writer.writeheader()

        review_writer = csv.DictWriter(
            review_fh,
            fieldnames=review_fields,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
        )
        review_writer.writeheader()

        for line_number, line in enumerate(
            tqdm(
                source,
                total=total_input_chunks,
                desc="Applying education gate v3",
                unit="chunk",
                dynamic_ncols=True,
            ),
            start=1,
        ):
            if not line.strip():
                continue

            record = json.loads(line)
            text = extract_text(record)
            chunk_id = extract_chunk_id(record, line_number)
            doc_id = extract_doc_id(record)

            total += 1
            docs_all.add(doc_id)

            anchor_hits = find_hits(text, anchors)
            contextual_hits = find_hits(text, contextual)
            tag_hits = find_hits(text, tag_only)

            anchor_groups = groups_set(anchor_hits)
            contextual_groups = groups_set(contextual_hits)
            tag_groups = groups_set(tag_hits)

            # Core gate rule: only source-backed ANCHORS determine inclusion.
            edu_match = bool(anchor_hits)
            edu_match_rule = "source_backed_anchor" if edu_match else "no_anchor"

            k12_signal = bool(anchor_groups & k12_groups)
            higher_ed_signal = bool(anchor_groups & higher_ed_groups)
            formal_vet_signal = "formal_vet" in anchor_groups

            pedagogy_signal = bool(
                anchor_groups & {"pedagogy", "k12_pedagogy"}
                or "pedagogy" in contextual_groups
            )

            teacher_development_signal = "teacher_development" in anchor_groups

            workforce_capacity_signal = bool(
                "workforce_capacity" in contextual_groups
                or "workforce_capacity" in tag_groups
            )

            higher_ed_research_signal = "higher_ed_research" in tag_groups

            row = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "edu_match": edu_match,
                "edu_match_rule": edu_match_rule,
                "anchor_groups": unique_join(anchor_hits, "group"),
                "anchor_canonicals": unique_join(anchor_hits, "canonical"),
                "anchor_terms": unique_join(anchor_hits, "term"),
                "contextual_groups": unique_join(contextual_hits, "group"),
                "contextual_canonicals": unique_join(contextual_hits, "canonical"),
                "contextual_terms": unique_join(contextual_hits, "term"),
                "tag_only_groups": unique_join(tag_hits, "group"),
                "tag_only_canonicals": unique_join(tag_hits, "canonical"),
                "tag_only_terms": unique_join(tag_hits, "term"),
                "k12_explicit_signal": k12_signal,
                "higher_ed_explicit_signal": higher_ed_signal,
                "formal_vet_signal": formal_vet_signal,
                "pedagogy_signal": pedagogy_signal,
                "teacher_development_signal": teacher_development_signal,
                "workforce_capacity_signal": workforce_capacity_signal,
                "higher_ed_research_signal": higher_ed_research_signal,
            }

            decision_writer.writerow(row)

            if not edu_match:
                continue

            kept += 1
            docs_kept.add(doc_id)

            for hit in anchor_hits:
                anchor_group_counts[hit["group"]] += 1
                anchor_canonical_counts[hit["canonical"]] += 1
                anchor_term_counts[hit["term"]] += 1

            for hit in contextual_hits:
                contextual_group_counts[hit["group"]] += 1

            for hit in tag_hits:
                tag_group_counts[hit["group"]] += 1

            if k12_signal:
                subgroup_counts["k12_explicit"] += 1
            if higher_ed_signal:
                subgroup_counts["higher_ed_explicit"] += 1
            if formal_vet_signal:
                subgroup_counts["formal_vet"] += 1
            if pedagogy_signal:
                subgroup_counts["pedagogy"] += 1
            if teacher_development_signal:
                subgroup_counts["teacher_development"] += 1
            if workforce_capacity_signal:
                subgroup_counts["workforce_capacity"] += 1
            if higher_ed_research_signal:
                subgroup_counts["higher_ed_research"] += 1
            if k12_signal and higher_ed_signal:
                subgroup_counts["k12_and_higher_ed"] += 1

            enriched = dict(record)
            enriched.update({
                "edu_gate_version": version,
                "edu_gate_status": status,
                "edu_match": True,
                "edu_match_rule": edu_match_rule,
                "edu_anchor_hits": anchor_hits,
                "edu_contextual_hits": contextual_hits,
                "edu_tag_only_hits": tag_hits,
                "k12_explicit_signal": k12_signal,
                "higher_ed_explicit_signal": higher_ed_signal,
                "formal_vet_signal": formal_vet_signal,
                "pedagogy_signal": pedagogy_signal,
                "teacher_development_signal": teacher_development_signal,
                "workforce_capacity_signal": workforce_capacity_signal,
                "higher_ed_research_signal": higher_ed_research_signal,
            })

            chunks_fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            review_row = dict(row)
            review_row["text"] = WS_RE.sub(" ", str(text)).strip()
            review_writer.writerow(review_row)

    with summary_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["gate_version", version])
        writer.writerow(["gate_status", status])
        writer.writerow(["input_chunks", total])
        writer.writerow(["education_chunks", kept])
        writer.writerow(["education_chunk_share", round(kept / total, 8) if total else 0])
        writer.writerow(["input_documents", len(docs_all)])
        writer.writerow(["education_documents", len(docs_kept)])
        writer.writerow([
            "education_document_share",
            round(len(docs_kept) / len(docs_all), 8) if docs_all else 0,
        ])

        for name, count in subgroup_counts.most_common():
            writer.writerow([f"subgroup::{name}", count])

        for name, count in anchor_group_counts.most_common():
            writer.writerow([f"anchor_group::{name}", count])

        for name, count in anchor_canonical_counts.most_common():
            writer.writerow([f"anchor_canonical::{name}", count])

        for name, count in anchor_term_counts.most_common():
            writer.writerow([f"anchor_term::{name}", count])

    manifest = {
        "gate_version": version,
        "gate_status": status,
        "construct": lexicon.get("construct", {}),
        "selection_principles": lexicon.get("selection_principles", []),
        "source_framework": lexicon.get("source_framework", {}),
        "input_path": str(infile),
        "input_sha256": sha256(infile),
        "lexicon_path": str(lexicon_path),
        "lexicon_sha256": sha256(lexicon_path),
        "input_chunks": total,
        "education_chunks": kept,
        "input_documents": len(docs_all),
        "education_documents": len(docs_kept),
        "core_inclusion_rule": "one_or_more_source_backed_anchor_terms",
        "contextual_terms_affect_inclusion": False,
        "tag_only_terms_affect_inclusion": False,
    }

    manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("EDUCATION GATE V3 COMPLETE")
    print("--------------------------")
    print("Gate version:", version)
    print("Gate status:", status)
    print("Input chunks:", total)
    print("Education chunks:", kept)
    print("Education chunk share:", f"{100 * kept / total:.2f}%" if total else "0.00%")
    print("Education documents:", len(docs_kept), "of", len(docs_all))

    print("\nSUBGROUP SIGNALS AMONG INCLUDED CHUNKS")
    for name, count in subgroup_counts.most_common():
        pct = 100 * count / kept if kept else 0
        print(f"{name:28} {count:6d}  {pct:6.2f}%")

    print("\nTOP ANCHOR GROUPS")
    for name, count in anchor_group_counts.most_common():
        print(f"{name:28} {count:6d}")

    print("\nTOP ANCHOR TERMS")
    for name, count in anchor_term_counts.most_common(30):
        print(f"{name:40} {count:6d}")

    print("\nOUTPUTS")
    print(chunks_out)
    print(decisions_out)
    print(review_out)
    print(summary_out)
    print(manifest_out)


if __name__ == "__main__":
    main()
