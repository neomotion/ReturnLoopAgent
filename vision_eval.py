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
