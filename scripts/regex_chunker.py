import argparse, json, re, os

def normalize(text):
    # join hyphenated line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # normalize newlines
    text = re.sub(r"\r\n", "\n", text)
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

BOUNDARY_RE = re.compile(
    r"^(Article|ART\.?|Art\.?|Chapter|CHAPTER|Part|PART|Section|SECTION|Annex|ANNEX)\b"
    r"|^\(?\d+\)?[.)]\s+"
    r"|^\(?[a-z]\)?[.)]\s+"
    r"|^\(?[ivxlcdm]+\)?[.)]\s+",
    re.I
)

def chunk_text(doc_id, text):
    chunks = []
    current = []
    idx = 0

    for line in text.split("\n"):
        if BOUNDARY_RE.match(line.strip()) and current:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}__chunk{idx:05d}",
                "chunk_text": "\n".join(current).strip()
            })
            idx += 1
            current = []
        current.append(line)

    if current:
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}__chunk{idx:05d}",
            "chunk_text": "\n".join(current).strip()
        })

    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.output, "w", encoding="utf-8") as out:
        for fn in sorted(os.listdir(args.input_dir)):
            if not fn.endswith(".txt"):
                continue
            doc_id = fn.replace(".txt","")
            text = open(os.path.join(args.input_dir, fn), encoding="utf-8").read()
            text = normalize(text)
            for c in chunk_text(doc_id, text):
                out.write(json.dumps(c, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

