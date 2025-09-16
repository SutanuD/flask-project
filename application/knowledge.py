from pathlib import Path
import os, re, json, hashlib
import fitz  # PyMuPDF
from flask import current_app
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

KB_DIR       = Path(current_app.root_path) / "content" / "kb"
PDF_DIR      = Path(current_app.root_path) / "content" / "pdfs"
META_JSONL   = f"{KB_DIR}/meta.jsonl"
META_TMP     = f"{KB_DIR}/meta.tmp.jsonl"
INDEX_PATH   = f"{KB_DIR}/index.faiss"
BM25_PATH    = f"{KB_DIR}/bm25.json"
SUBJECTS_JSON= f"{KB_DIR}/subjects.json"
TOPICS_JSON = "/content/kb/topics.json"
PROGRESS_JSON = "/content/kb/progress.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SUBJECT_ALIASES = {
    "science": ["science","physics","chemistry","biology","bio"],
    "mathematics": ["math","mathematics","algebra","geometry","trigonometry","statistics","probability","arith"],
    "english": ["english","literature","grammar","poem","prose"],
    "social science": ["social science","history","geography","civics","political science","economics","democratic","resources"],
    "hindi": ["hindi"],
    "computer science": ["computer","informatics","cs","ip","information technology","it"],
    "biology": ["biology","bio","life processes","cell","heredity","evolution","reproduction"],
    "physics": ["physics","electricity","magnetism","motion","force","work","energy","light","sound","wave"],
    "chemistry": ["chemistry","chemical","reaction","matter","atom","mole","periodic","compound","mixture"]
}

def _safe_text(s: str) -> str:
    if not s:
        return ""
    # strip NULs & normalize weird whitespace; keep \n
    s = s.replace("\x00", "")
    s = s.encode("utf-8", "replace").decode("utf-8", "replace")
    # collapse repeated spaces while preserving newlines
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    # remove stray control chars (except \n and \t)
    s = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", s)
    return s.strip()

def parse_grade_subject_from_filename(fname: str):
    chapter = os.path.splitext(fname)[0].split("_")[2].lower()
    base = os.path.splitext(fname)[0].replace("_"," ").lower()
    m = re.search(r"(class|grade)\s*([0-9]{1,2})", base)
    grade = int(m.group(2)) if m else None
    subject = None
    for subj, keys in SUBJECT_ALIASES.items():
        if any(k in base for k in keys):
            subject = subj; break
    if subject is None:
        m2 = re.search(r"([a-z]+)", base)
        subject = m2.group(1) if m2 else "general"
    return grade, subject, chapter

def iter_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text") or ""
        yield pno+1, _safe_text(text)

def heading_guess(page_text):
    lines = [l.strip() for l in page_text.split("\n") if l.strip()]
    chap, title = "", ""
    for i,l in enumerate(lines[:10]):
        if re.match(r"(?i)^chapter\s+\d+\b", l):
            chap = l
            if i+1 < len(lines): title = lines[i+1][:120]
            break
    if not chap and lines:
        title = lines[0][:120]
    return chap, title

def chunk_page(text, max_chars=1200, overlap=100):
    # split on blank lines, rebuild chunks under max_chars
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf: chunks.append(buf)
            buf = p
    if buf: chunks.append(buf)
    out = []
    for i,c in enumerate(chunks):
        prev_tail = chunks[i-1][-overlap:] if i>0 else ""
        out.append((prev_tail + "\n" + c).strip())
    return out

def embed_texts(texts):
    # OpenAI embeddings; normalized for cosine via IP
    B = 512
    vecs = []
    for i in range(0, len(texts), B):
        resp = client.embeddings.create(model="text-embedding-3-large", input=texts[i:i+B])
        arr = np.array([d.embedding for d in resp.data], dtype="float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        vecs.append(arr.astype("float32"))
    return np.vstack(vecs)

def build_kb(pdf_dir=PDF_DIR):
    os.makedirs(KB_DIR, exist_ok=True)

    metadatas, texts = [], []
    subjects_registry = {}

    # Collect chunks + metadata
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        book = os.path.splitext(fname)[0]
        grade, subject, ch = parse_grade_subject_from_filename(fname)
        subjects_registry.setdefault(subject, set()).add(grade)

        for page_no, page_text in iter_pdf_pages(path):
            chap, sec = heading_guess(page_text)
            for c in chunk_page(page_text):
                metadatas.append({
                    "book": book,
                    "grade": grade,
                    "subject": subject,
                    "chapter": ch,
                    "section": sec,
                    "page": page_no
                })
                texts.append(c)

    if not texts:
        raise RuntimeError("No chunks created. Did you upload PDFs?")

    # Embeddings + FAISS
    print(f"Embedding {len(texts)} chunks…")
    X = embed_texts(texts)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, INDEX_PATH)


    with open(META_TMP, "w", encoding="utf-8", newline="\n") as f:
        for m, t in zip(metadatas, texts):
            m2 = {**m, "text": t}
            f.write(json.dumps(m2, ensure_ascii=False) + "\n")

    if os.path.exists(META_JSONL):
        os.remove(META_JSONL)
    os.rename(META_TMP, META_JSONL)


    valid_count = 0
    with open(META_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s: continue
            try:
                json.loads(s)
                valid_count += 1
            except json.JSONDecodeError as e:
                snippet = s[max(0, e.pos-80):e.pos+80]
                raise RuntimeError(f"Invalid JSON at line {i}: {e.msg} (pos {e.pos}).\nSnippet: {snippet}") from e
    print(f"meta.jsonl written & validated ({valid_count} lines).")


    tokenized = [t.lower().split() for t in texts]
    bm25 = {"docs": texts, "tokenized": tokenized}
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25, f)


    sr = {k: sorted([g for g in v if g is not None]) for k,v in subjects_registry.items()}
    with open(SUBJECTS_JSON, "w", encoding="utf-8") as f:
        json.dump(sr, f, indent=2)

    print(f"KB ready: {len(texts)} chunks; FAISS+BM25 built.")
    print("Detected subjects/grades:", sr)

#call from endpoint
#build_kb()