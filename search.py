import json
import re
from pathlib import Path

import numpy as np
from model2vec import StaticModel
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "data"
STOPWORDS = {"a", "an", "the", "is", "are", "to", "of", "for", "and", "or", "in", "on", "it", "my",
             "i", "can", "do", "how", "what", "when", "will", "with", "from", "but", "has", "not"}

# ---- chunk the policy docs: one chunk per "## " section, with the doc title in front ----
chunks = []
for path in sorted((DATA / "policies").glob("*.md")):
    doc_title, *sections = path.read_text(encoding="utf-8").split("\n## ")
    for section in sections:
        title, text = section.split("\n", 1)
        chunks.append({"id": f"{path.stem}#{title.lower().replace(' ', '-')}",
                       "text": f"{doc_title.strip('# ')} > {title}: {text.strip()}"})

def stem(w):
    # crude stemmer: returns -> return, changed / change -> chang
    for end in ["ing", "ed", "es", "e", "s"]:
        if w.endswith(end) and len(w) > len(end) + 3:
            return w[:-len(end)]
    return w

def words(text):
    # lowercase, keep 20,000 as one number, drop punctuation and stopwords
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower().replace(",", "")) if w not in STOPWORDS]

# ---- build both indexes ----
bm25 = BM25Okapi([words(c["text"]) for c in chunks])
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
vectors = model.encode([c["text"] for c in chunks])
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

def bm25_rank(query):
    return list(np.argsort(-bm25.get_scores(words(query))))

def dense_rank(query):
    v = model.encode([query])[0]
    return list(np.argsort(-(vectors @ (v / np.linalg.norm(v)))))

def hybrid_rank(query, k=60):
    # reciprocal rank fusion: only ranks matter, so the two score scales don't need to match
    scores = {}
    for ranking in [bm25_rank(query), dense_rank(query)]:
        for rank, i in enumerate(ranking):
            scores[i] = scores.get(i, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

def search(query, top_k=3):
    return [chunks[i] for i in hybrid_rank(query)[:top_k]]

# ---- eval: recall@3 and MRR on labelled questions ----
if __name__ == "__main__":
    questions = json.loads((DATA / "policy_questions.json").read_text(encoding="utf-8"))
    print(f"{len(chunks)} chunks, {len(questions)} questions\n")
    for name, rank_fn in [("bm25", bm25_rank), ("dense", dense_rank), ("hybrid", hybrid_rank)]:
        ranks = [[chunks[i]["id"] for i in rank_fn(x["q"])].index(x["gold"]) + 1 for x in questions]
        recall = sum(r <= 3 for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks) / len(ranks)
        misses = [x["q"] for x, r in zip(questions, ranks) if r > 3]
        print(f"{name:7} recall@3 {recall:.2f}  MRR {mrr:.2f}  misses: {misses}")
