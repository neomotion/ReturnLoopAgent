import base64
import json
import sys
import time
from pathlib import Path

from vision_data import OUT, PROMPT

def parse(text):
    # pull the JSON out of a model's reply; a reply we can't read counts as all False
    text = text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def score(rows, preds):
    # damage precision: a false "damaged" lets a fake claim through the photo check
    # damage recall: a missed defect sends a genuine claim to human review (policy), costing reviewer time
    tp = sum(r["shows_damage"] and p.get("shows_damage") is True for r, p in zip(rows, preds))
    fp = sum(not r["shows_damage"] and p.get("shows_damage") is True for r, p in zip(rows, preds))
    fn = sum(r["shows_damage"] and p.get("shows_damage") is not True for r, p in zip(rows, preds))
    product_ok = sum(r["shows_product"] == p.get("shows_product") for r, p in zip(rows, preds))
    unreadable = sum(not p for p in preds)
    return {"damage_precision": round(tp / max(tp + fp, 1), 3), "damage_recall": round(tp / max(tp + fn, 1), 3),
            "damage_f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
            "product_accuracy": round(product_ok / len(rows), 3), "unreadable": unreadable}

def apply_threshold(preds, probs, t):
    # "damaged" when the model's probability of answering true is at least t
    return [p | {"shows_damage": prob >= t} for p, prob in zip(preds, probs)]

def pick_threshold(rows, preds, probs, min_precision=0.95):
    # highest recall that keeps damage precision >= min_precision (fake claims stay out)
    best, best_recall = 0.5, -1
    for t in [i / 100 for i in range(5, 96)]:
        s = score(rows, apply_threshold(preds, probs, t))
        if s["damage_precision"] >= min_precision and s["damage_recall"] > best_recall:
            best, best_recall = t, s["damage_recall"]
    return best

def recall_by_defect(rows, preds):
    # which kinds of damage the model misses
    found = {}
    for r, p in zip(rows, preds):
        if r["shows_damage"]:
            hit, total = found.get(r["defect"], (0, 0))
            found[r["defect"]] = (hit + (p.get("shows_damage") is True), total + 1)
    return {d: f"{hit}/{total}" for d, (hit, total) in sorted(found.items(), key=lambda x: x[1][0] / x[1][1])}

def gemini(row):
    from llm_config import client, MODEL   # imported here so Colab can use score() without a Gemini key
    image = base64.b64encode((OUT / row["image"]).read_bytes()).decode()
    reply = client.chat.completions.create(model=MODEL, temperature=0, messages=[{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(product=row["product"])},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]}])
    return reply.choices[0].message.content

# ---- run Gemini on the test set: py vision_eval.py [how many] ----
if __name__ == "__main__":
    rows = [json.loads(line) for line in (OUT / "test.jsonl").read_text().splitlines()]
    rows = rows[:int(sys.argv[1])] if len(sys.argv) > 1 else rows
    preds, start = [], time.time()
    for i, row in enumerate(rows):
        preds.append(parse(gemini(row)))
        print(f"{i + 1}/{len(rows)} {row['category']}/{row['defect']} -> {preds[-1]}")
    result = score(rows, preds) | {"seconds_per_photo": round((time.time() - start) / len(rows), 2)}
    (OUT / "result_gemini.json").write_text(json.dumps(result, indent=1))
    print(result)
