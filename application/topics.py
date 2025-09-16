from pathlib import Path
import os, re, json, math, collections
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from json import JSONDecoder, JSONDecodeError # Import for load_meta_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "content" / "kb"
# --- Robust meta loader ---
def load_meta_jsonl(path=KB_DIR / "meta.jsonl"):
    """
    Loads /content/kb/meta.jsonl robustly:
      - Handles proper JSONL (one JSON per line)
      - Also handles concatenated JSON without newlines (}{ back-to-back)
      - Strips NULs and ignores empty/whitespace segments
    """
    # Placeholder: using the logic from cell tZnxJ19J79y-
    dec = JSONDecoder()
    objs = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        buf = f.read()

    # Remove NULs that may appear from odd PDFs
    buf = buf.replace("\x00", "")

    # Fast path: try line-by-line first (proper JSONL)
    lines = buf.splitlines()
    if len(lines) > 1:
        for ln, line in enumerate(lines, 1):
            s = line.strip()
            if not s:
                continue
            try:
                objs.append(json.loads(s))
            except JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line {ln} in {path}: {e}")
                continue
        return objs

    # Streaming decode: parse sequential JSON objects in one big string
    idx = 0
    n = len(buf)
    while idx < n:
        # skip whitespace between JSONs
        while idx < n and buf[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(buf, idx)
        except JSONDecodeError as e:
            # Try to repair the most common issue: missing newline between }{
            # Insert a newline at the nearest }{}{ boundary near the error and continue once.
            window = buf[max(0, idx-50):min(n, idx+50)]
            if "}{".encode().decode() in window:
                buf = buf.replace("}{", "}\n{")
                # Restart the whole streaming pass once after patch
                objs.clear()
                idx = 0
                n = len(buf)
                continue
            # If still bad, print warning and try to skip the character
            context = buf[max(0, idx-140):min(n, idx+140)]
            print(f"Warning: Skipping invalid JSON segment at pos {idx} in {path}. Context: {context}. Error: {e}")
            idx += 1 # Skip one character and try again
            continue # Move to the next iteration


        objs.append(obj)
        idx = end
    return objs


META_PATH = KB_DIR / "meta.jsonl"
assert os.path.exists(META_PATH), "meta.jsonl not found. Build the KB first."

# Load fresh each time to reflect any new ingest
def get_meta_rows():
    return load_meta_jsonl(META_PATH)


# --- Utilities for topic extraction ---
CUE_PATTERNS = [
    r"\b(key points?|summary|in a nutshell|important|remember|definition|define|note|quick recap)\b",
    r"\b(exercise|questions|mcq|short answer|long answer|try yourself)\b",
    r"\b(activity\s*\d*|project|experiment|investigate)\b",
    r"\b(objectives?|learning outcomes?)\b",
    r"\b(fascinating facts?|holistic lens|know a scientist)\b",
]
CUE_RE = re.compile("|".join(CUE_PATTERNS), re.I)

HEADING_LINE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,3})\s*[:\-\)]?\s*([A-Z][^\n]{3,200})$")

# broad stopwords; keep domain-neutral
GENERIC_STOP = set("""chapter exercise question questions answer answers points summary topic topics figure table example examples
activity project experiment objectives outcome outcomes learning page pages section subsection student students teacher teachers
colour color paper group let us observe discuss try find out make list see also aim material apparatus method conclusion
""".split())

def compress_ranges(pages: List[int]) -> str:
    if not pages: return ""
    pages = sorted(set(pages))
    rng=[]; s=prev=pages[0]
    for p in pages[1:]:
        if p==prev+1: prev=p
        else: rng.append((s,prev)); s=prev=p
    rng.append((s,prev))
    return ",".join([f"{a}-{b}" if a!=b else f"{a}" for a,b in rng])

def extract_chapter_and_book(rows, subject, grade, book, chapter):
    ch_blobs, ch_pages = [], []
    book_blobs = []
    for r in rows:
        if r.get("subject")==subject and r.get("grade")==grade:
            book_blobs.append(r.get("text",""))
            if (r.get("chapter") or "") == chapter:
                ch_blobs.append(r.get("text",""))
                ch_pages.append(r.get("page",0))
    return ch_blobs, ch_pages, book_blobs

def split_lines(text):
    return [ln.strip() for ln in text.split("\n") if ln.strip()]

def cue_heading_boosts(blobs: List[str]):
    term_boost = collections.Counter()
    page_hits = collections.defaultdict(set)

    def norm_phrase(s):
        s = re.sub(r"[^a-z0-9\s\-]", " ", s.lower())
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for bi, t in enumerate(blobs):
        for ln in split_lines(t):
            m = HEADING_LINE_RE.match(ln)
            if m:
                phrase = norm_phrase(m.group(1))
                if phrase and phrase not in GENERIC_STOP and not phrase.isdigit():
                    term_boost[phrase] += 2.0
                    page_hits[phrase].add(bi)
                continue
            if CUE_RE.search(ln):
                phrase = norm_phrase(ln)
                term_boost[phrase] += 1.0
                page_hits[phrase].add(bi)

        # Capitalized multiword spans
        for m in re.finditer(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6})", t):
            phrase = norm_phrase(m.group(0))
            if phrase and len(phrase.split())>=2:
                term_boost[phrase] += 0.6
                page_hits[phrase].add(bi)

    return term_boost, page_hits

def clean_phrase(p):
    # keep alphabetic + hyphen; remove repeated spaces; drop very short tokens
    p = re.sub(r"[^a-z\s\-]", " ", p.lower())
    p = re.sub(r"\s+", " ", p).strip()
    # drop phrases containing generic stopwords entirely
    toks = p.split()
    if any(tok in GENERIC_STOP for tok in toks):
        return None
    # keep multiword phrases or domainy unigrams (litmus, neutralisation, indicator)
    if len(toks)==1 and toks[0] not in {"litmus","indicator","neutralisation","acidic","basic","neutral"}:
        return None
    # avoid phrases starting with verbs like "let", "make", "see"
    if toks and toks[0] in {"let","make","see","find","observe","discuss","try"}:
        return None
    # min length
    if len(" ".join(toks)) < 5:
        return None
    return " ".join(toks)

def tfidf_phrases(ch_blobs: List[str], book_blobs: List[str], top_k=40):
    # Chapter-vs-book TFIDF with 2-4 grams to prefer phrases
    # Relax min_df and max_df to handle smaller chapter texts or more unique terms
    vect = TfidfVectorizer(
        ngram_range=(2,4),
        min_df=0.01,  # Lowered from default 1 -> Adjusted again
        max_df=0.99,  # Increased from default 0.9 -> Adjusted again
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"
    )
    corpus = ["\n".join(ch_blobs)] + ["\n".join(book_blobs)]  # doc0=chapter, doc1+=book
    try:
        X = vect.fit_transform(corpus)
    except ValueError as e:
        if "After pruning, no terms remain" in str(e):
            print(f"Warning: TF-IDF found no terms for chapter. Consider adjusting min_df/max_df further or check input text.")
            return [] # Return empty list instead of raising error
        else:
            raise # Re-raise other ValueErrors
    ch_vec = X[0].toarray()[0]
    feats = vect.get_feature_names_out()
    pairs = [(feats[i], ch_vec[i]) for i in ch_vec.nonzero()[0]]
    pairs.sort(key=lambda x: x[1], reverse=True)
    out=[]
    for term, sc in pairs[:max(top_k, 60)]:
        cp = clean_phrase(term)
        if cp:
            out.append((cp, float(sc)))
    return out

def pages_for_phrase(phrase: str, blobs: List[str]):
    hits=[]
    pat = re.compile(r"\b" + re.escape(phrase) + r"\b", re.I)
    for i, t in enumerate(blobs):
        if pat.search(t):
            hits.append(i)
    return hits

def robust_tfidf_terms(ch_blobs: List[str], book_blobs: List[str], want=60):
    corpus = ["\n".join(ch_blobs)] + ["\n".join(book_blobs)]
    attempts = [
        dict(ngram_range=(2,4), stop_words="english", max_df=0.95, min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"),
        dict(ngram_range=(1,3), stop_words=None,      max_df=1.0,  min_df=1, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"),
        dict(ngram_range=(1,3), stop_words=None,      max_df=1.0,  min_df=1, token_pattern=r"(?u)\b\w[\w\-]+\b"),
    ]
    for cfg in attempts:
        try:
            vect = TfidfVectorizer(**cfg)
            X = vect.fit_transform(corpus)
            if X.shape[1] == 0:
                continue
            ch_vec = X[0].toarray()[0]
            feats = vect.get_feature_names_out()
            # keep top features present in chapter doc
            idxs = ch_vec.nonzero()[0]
            pairs = [(feats[i], ch_vec[i]) for i in idxs]
            pairs.sort(key=lambda x: x[1], reverse=True)
            out=[]
            for term, sc in pairs[:max(want, 40)]:
                cp = clean_phrase(term)
                if cp:
                    out.append((cp, float(sc)))
            if out:
                return out
        except ValueError:
            continue
    # if everything fails, return empty and let cue/heading logic handle it
    return []

def merge_scores(tfidf_terms, boost_counter, boost_pages, blobs):
    cand = {}
    for p, s in tfidf_terms:
        cand[p] = {"score": s, "blob_indices": pages_for_phrase(p, blobs)}

    for raw, b in boost_counter.items():
        p = clean_phrase(raw)
        if not p: continue
        hits = boost_pages.get(raw, set())
        hits = [h for h in hits if h < len(blobs)]
        c = cand.setdefault(p, {"score": 0.0, "blob_indices": hits})
        c["score"] += float(b) * 0.2
        c["blob_indices"] = sorted(set(c["blob_indices"]) | set(hits))

    for p, c in cand.items():
        cov = len(set(c["blob_indices"]))
        c["score"] *= (1.0 + 0.05 * cov)
    return cand

def pick_topics(cand: Dict[str, Dict[str, Any]], blobs: List[str], top_n: int):
    items = sorted(cand.items(), key=lambda kv: kv[1]["score"], reverse=True)
    picked=[]
    for term, info in items:
        if any(term in t for t,_ in picked if len(t) > len(term)+2):
            continue
        picked.append((term, info))
        if len(picked) >= top_n*2:
            break
    final=[]
    for term, info in picked[:top_n]:
        idxs = info["blob_indices"]
        words = sum(len(blobs[i].split()) for i in idxs if i < len(blobs))
        est = max(6, math.ceil(words/260.0) + 3)
        final.append({
            "topic": " ".join(w.capitalize() if i==0 else w for i,w in enumerate(term.split())),
            "score": round(info["score"], 4),
            "blob_indices": idxs,
            "estimated_minutes": int(est)
        })
    return final


def _llm_explain(topics, subject, grade, book, chapter):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except Exception:
        return topics
    if not topics:
        return topics
    bullets = "\n".join(f"- {t['topic']}" for t in topics)
    system = ("You help a student prioritize a chapter. For each topic name, "
              "write ONE concise sentence on why it's important within the chapter scope. "
              "Return JSON list of {topic, why_important}. No new topics.")
    user = f"Subject: {subject}, Grade: {grade}, Book: {book}\nChapter: {chapter}\nTopics:\n{bullets}\n"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)
        mp = {d["topic"].strip().lower(): d["why_important"] for d in data if "topic" in d and "why_important" in d}
        for t in topics:
            k = t["topic"].lower()
            if k in mp:
                t["why_important"] = mp[k][:240]
    except Exception:
        pass
    return topics