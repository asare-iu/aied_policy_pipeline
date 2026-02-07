#!/usr/bin/env python3
import argparse
import pandas as pd

def split_terms(s: str):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # your file uses "; " as separator
    return [t.strip() for t in s.split(";") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_terms_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit_per_side", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.top_terms_csv)

    rows = []
    for _, r in df.iterrows():
        pc = str(r["pc"])
        pos = split_terms(r.get("top_positive_terms", ""))
        neg = split_terms(r.get("top_negative_terms", ""))

        pos = pos[:args.limit_per_side]
        neg = neg[:args.limit_per_side]

        for rank, term in enumerate(pos, start=1):
            rows.append({"pc": pc, "side": "pos", "rank": rank, "term": term})
        for rank, term in enumerate(neg, start=1):
            rows.append({"pc": pc, "side": "neg", "rank": rank, "term": term})

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    main()
