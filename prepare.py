#Builds the index: reads PDFs -> cleans text -> cuts chunks -> embeds -> saves
import argparse
import datetime
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

DOC_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "
PAGE_NO = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)


def pattern(line):
    #"Page 3 of 40" and "Page 4 of 40" become the same pattern
    return re.sub(r"\d+", "#", line.lower())


def join_lines(lines):
    text = ""
    for line in lines:
        if text.endswith("-") and text[-2:-1].isalpha() and line[:1].islower():
            text = text[:-1] + line  # "re-" + "taking" -> "retaking"
        else:
            text += " " + line
    return re.sub(r"\s+", " ", text).strip()


def clean_document(pages):
    #pages = raw text of every page of one PDF. Returns cleaned pages + removed header/footer patterns
    all_lines = []
    for text in pages:
        text = unicodedata.normalize("NFKC", text)  #ligatures (fi), odd spaces
        text = text.replace("\u00ad", "")  #soft hyphens
        text = re.sub(r"[\u2010-\u2012]", "-", text)  #fancy hyphens -> "-"
        text = re.sub(r"[\x00-\x08\x0b-\x1f\u200b-\u200f\ufeff]", " ", text)  #control / zero-width characters
        text = re.sub(r"(?:\s?[.\u00b7_]){4,}", " ", text)  #dot leaders in contents lists
        all_lines.append([l.strip() for l in text.splitlines() if l.strip()])

    #a header/footer = a line pattern at the top or bottom of at least 40% of the pages
    seen = Counter()
    for lines in all_lines:
        seen.update({pattern(l) for l in lines[:3] + lines[-3:]})
    repeated = {p for p, n in seen.items() if n >= max(3, 0.4 * len(pages))}

    cleaned, removed = [], Counter()
    for lines in all_lines:
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < 3 or i >= len(lines) - 3
            if at_edge and (pattern(line) in repeated or PAGE_NO.match(line)):
                removed[pattern(line)] += 1
            else:
                keep.append(line)
        cleaned.append(join_lines(keep))
    return cleaned, removed


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)  #ids are cited in [brackets]: no spaces


def make_chunks(doc, page, text, size, overlap):
    w, step = text.split(), size - overlap
    for n, start in enumerate(range(0, len(w), step), 1):
        body = " ".join(w[start:start + size])
        yield {"id": f"{safe_name(doc)}:p{page}:c{n}", "document": doc,
               "position": f"page {page}, chunk {n}",
               "lang": "lt" if re.search(r"[ąčęėįšųūž]", body.lower()) else "en", "text": body}
        if start + size >= len(w):  #this window reached the end of the page
            break


def main():
    ap = argparse.ArgumentParser(description="Building the RAG index from PDFs.")
    ap.add_argument("--docs", default="docs", help="folder with the PDFs")
    ap.add_argument("--embed-model", default="embeddinggemma")
    ap.add_argument("--chunk-words", type=int, default=80)
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--min-words", type=int, default=15, help="skip pages with fewer words after cleaning")
    ap.add_argument("--index", default="index.npz")
    ap.add_argument("--chunks", default="corpus/chunks.json")
    ap.add_argument("--batch", type=int, default=32, help="chunks per embedding request")
    ap.add_argument("--dry-run", action="store_true", help="only cleans and chunks; writes corpus/preview.txt")
    a = ap.parse_args()
    if a.overlap >= a.chunk_words:
        raise SystemExit("--overlap must be smaller than --chunk-words")
    pdfs = sorted(Path(a.docs).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {Path(a.docs).resolve()}")

    #Reads, cleans and chunks every PDF
    chunks, preview = [], []
    for path in pdfs:
        raw = []
        for page in PdfReader(path).pages:
            try:
                raw.append(page.extract_text() or "")
            except Exception:  #a broken page is skipped, not fatal
                raw.append("")
        pages, removed = clean_document(raw)
        kept = [(n, t) for n, t in enumerate(pages, 1) if len(t.split()) >= a.min_words]
        mine = [c for n, t in kept for c in make_chunks(path.name, n, t, a.chunk_words, a.overlap)]
        chunks += mine
        sample = sorted({0, 1, 2, len(mine) // 2} & set(range(len(mine))))  #first chunks + one from the middle
        preview += [f"=== {path.name}"] + [f"[{mine[i]['id']}] {mine[i]['text']}\n" for i in sample]
        print(f"{path.name}: {len(raw)} pages, {len(kept)} with text, "
              f"{sum(removed.values())} header/footer lines removed, {len(mine)} chunks")
        if removed:
            print("   removed patterns: " + ", ".join(f"'{p}' x{n}" for p, n in removed.most_common(3)))
        if not kept:
            print("   No readable text found. Is the PDF a scan?")
    if not chunks:
        raise SystemExit("No text found in any PDF.")
    ids = [c["id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate chunk ids: two files share a safe name. Rename one.")
    print(f"CHUNKED into {len(chunks)} passages ({a.chunk_words} words, overlap {a.overlap})")

    Path(a.chunks).parent.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        out = Path(a.chunks).parent / "preview.txt"
        out.write_text("\n".join(preview), encoding="utf-8")
        raise SystemExit(f"Dry run: read {out} to check the extracted text. Nothing embedded.")

    #Embeds in batches; unit-length vectors so a dot product equals cosine similarity
    vectors = []
    for i in range(0, len(chunks), a.batch):
        batch = [DOC_PREFIX + c["text"] for c in chunks[i:i + a.batch]]
        vectors += ollama.embed(model=a.embed_model, input=batch, truncate=False).embeddings
        print(f"  embedded {min(i + a.batch, len(chunks))}/{len(chunks)}", end="\r")
    vectors = np.array(vectors, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    print(f"\nINDEX shape: {vectors.shape} (chunks, embedding dimensions)")

    #Saves the index and the readable chunk list
    Path(a.index).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.index, vectors=vectors, ids=np.array(ids), texts=np.array([c["text"] for c in chunks]),
                        embedding_model=a.embed_model, query_prefix=QUERY_PREFIX, doc_prefix=DOC_PREFIX)
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "embedding_model": a.embed_model, "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX,
            "dimensions": int(vectors.shape[1]), "documents": [p.name for p in pdfs],
            "chunking": {"words": a.chunk_words, "overlap": a.overlap, "min_page_words": a.min_words,
                         "never_crosses": "page"},
            "cleaning": ["NFKC", "soft/fancy hyphens", "control chars", "dot leaders",
                         "repeated header/footer lines", "page numbers", "hyphen line breaks"],
            "chunks": chunks}
    Path(a.chunks).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"SAVED {a.index} and {a.chunks}")


if __name__ == "__main__":
    main()
