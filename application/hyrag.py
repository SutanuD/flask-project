from flask import current_app
import json, os, numpy as np, faiss, math, re
from pathlib import Path
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Resolve project root without requiring a Flask app context
PROJECT_ROOT = Path(current_app.root_path)
KB_DIR = PROJECT_ROOT / "content" / "kb"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Try to load the faiss index but don't allow a context error to stop module import

def load():
    try:
        index_path = KB_DIR / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        # faiss.read_index expects a filename string; ensure we pass a str
        index = faiss.read_index(str(index_path))
    except Exception as e:
        print("Failed loading FAISS index:", e)
        index = None

    meta = []
    try:
        with open(KB_DIR / "meta.jsonl", encoding="utf-8") as f:
            print("Loading metadata...", f.name)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta.append(json.loads(line))
    except Exception as e:
        print("Failed loading meta.jsonl:", e)

    try:
        with open(KB_DIR / "bm25.json", encoding="utf-8") as f:
            bm25_raw = json.load(f)
        bm25 = BM25Okapi([d for d in bm25_raw["tokenized"]])
    except Exception as e:
        print("Failed loading bm25.json:", e)
        bm25 = None

    try:
        with open(KB_DIR / "subjects.json", encoding="utf-8") as f:
            SUBJECTS = json.load(f)
    except Exception as e:
        print("Failed loading subjects.json:", e)
        SUBJECTS = {}

    # Return the loaded KB objects so they can be assigned at module level
    return index, meta, bm25, SUBJECTS


# Initialize module-level variables
index, meta, bm25, SUBJECTS = load()


def embed_query(q: str) -> np.ndarray:
    r = client.embeddings.create(model="text-embedding-3-large", input=[q])
    v = np.array(r.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def guess_subject(query: str):
    q = query.lower()
    best, best_score = None, 0
    for subj, grades in SUBJECTS.items():
        keys = subj.split() + sum(([k] for k,v in locals().get('SUBJECT_ALIASES', {}).items() if k==subj), [])
        score = sum(1 for k in keys if k in q)

        if subj in ("science","biology","physics","chemistry") and any(w in q for w in ["photosynthesis","cell","energy","acid","base","motion","light","electric","magnet"]):
            score += 1
        if subj in ("mathematics",) and any(w in q for w in ["theorem","solve","prove","equation","triangle","probability","mean","median","mode","lcm","hcf"]):
            score += 1
        if score > best_score:
            best, best_score = subj, score
    return best if best_score>0 else None

def hybrid_retrieve(question: str, subject: str=None, grade: int=None, k_embed=40, k_bm25=40, top=8):
    qv = embed_query(question)
    D, I = index.search(qv, k_embed)
    cand_embed = [(i, float(D[0][j])) for j,i in enumerate(I[0])]


    scores = bm25.get_scores(question.lower().split())
    top_bm25_ids = np.argsort(scores)[::-1][:k_bm25]
    cand_bm25 = [(int(i), float(scores[i])) for i in top_bm25_ids]


    cand_map = {}
    for i,s in cand_embed:
        cand_map.setdefault(i, {"embed":0.0,"bm25":0.0})
        cand_map[i]["embed"] = max(cand_map[i]["embed"], s)
    for i,s in cand_bm25:
        cand_map.setdefault(i, {"embed":0.0,"bm25":0.0})
        cand_map[i]["bm25"] = max(cand_map[i]["bm25"], s)


    if cand_map:
        max_bm25 = max(v["bm25"] for v in cand_map.values()) or 1.0
    fused = []
    for i,sc in cand_map.items():
        m = meta[i]
        subj_boost = 1.0
        if subject and m.get("subject")==subject:
            subj_boost *= 1.25
        if grade and m.get("grade")==grade:
            subj_boost *= 1.15

        txt = m["text"].lower()
        edu_boost = 1.1 if any(k in txt for k in ["definition","example","key points","summary","exercise"]) else 1.0
        f_score = 0.65*sc["embed"] + 0.35*(sc["bm25"]/max_bm25)
        fused.append((i, f_score*subj_boost*edu_boost))

    fused.sort(key=lambda x: x[1], reverse=True)
    picked = []
    seen_pages = set()
    for i, s in fused:
        m = meta[i]
        key = (m["book"], m["page"])
        if key in seen_pages:
            continue
        seen_pages.add(key)
        picked.append((m, s))
        if len(picked)>=top: break
    return picked

SYSTEM = (
    "You are a strict NCERT tutor. Use ONLY the provided context. "
    "Format your response as:\n"
    "1) Short definition/overview\n"
    "2) Key points (bulleted)\n"
    "3) Example or short derivation (if context contains one)\n"
    "4) Final takeaway\n"
    "Always include explicit NCERT citations after relevant paragraphs like (Book, Ch X, p.Y)."
)

def smart_answer(question: str, contexts_with_scores, dontknow_threshold=0.35):

    if not contexts_with_scores:
        return "I don’t know. I couldn’t find this in the NCERT materials you uploaded."
    avg_score = sum(s for _,s in contexts_with_scores)/len(contexts_with_scores)
    ctx_text = "\\n\\n---\\n\\n".join(
        f'[{c.get("subject","")}, {c.get("book","")}, {c.get("chapter","")}, p.{c.get("page","")}]\\n{c["text"][:1600]}'
        for c,_ in contexts_with_scores
    )
    if avg_score < dontknow_threshold:
        prefix = "I’m not fully confident this is covered in your NCERT set. Here’s my best effort using the closest matches.\\n\\n"
    else:
        prefix = ""
    user = f"{prefix}Context:\\n{ctx_text}\\n\\nQuestion: {question}\\nAnswer following the required format with citations."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":user}],
        temperature=0
    )
    return resp.choices[0].message.content

def smart_answer_ext(question: str, subject: str=None, grade: int=None):
    ext_prompt = (
        f"Provide broader, real-world context for this question beyond the NCERT syllabus. Keep it factual and suitable for {grade}-th grade student"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":ext_prompt},
            {"role":"user","content":question}
        ],
        temperature=0.7
    )
    return resp.choices[0].message.content

def chat_router(question: str, subject: str=None, grade: int=None):
    subj = subject or guess_subject(question)
    picks = hybrid_retrieve(question, subject=subj, grade=grade)
    answer = smart_answer(question, picks)
    additional = smart_answer_ext(question, subject=subj, grade=grade)
    cites = [{"book":m.get("book"),"subject":m.get("subject"),"chapter":m.get("chapter"),"page":m.get("page")} for m,_ in picks]
    return answer, cites, subj, additional
