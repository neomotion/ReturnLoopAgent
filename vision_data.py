import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

# ---- real damage photos: MVTec AD (CC BY-NC-SA 4.0), Voxel51's copy on Hugging Face ----
REPO = "Voxel51/mvtec-ad"
OUT = Path(__file__).parent / "data" / "vision"
PER_CLASS = 40   # up to 40 damaged + 40 good photos per product
TEST_SIZE = 200
PRODUCTS = {"bottle": "Glass bottle", "cable": "Power cable", "capsule": "Medicine capsule",
            "hazelnut": "Hazelnut", "metal_nut": "Metal nut", "pill": "Pill tablet", "screw": "Screw",
            "toothbrush": "Toothbrush", "transistor": "Transistor", "zipper": "Zipper"}

# the question we ask every model (Gemini, the base local model, the fine-tuned one)
PROMPT = """Ordered product: {product}. The customer says the item arrived damaged.
Look only at the photo. Does it show the ordered product? Does it show damage or a defect?
Answer only with JSON: {{"shows_product": true or false, "shows_damage": true or false}}"""

def answer(row):
    # the target the fine-tuned model learns to write
    return json.dumps({"shows_product": row["shows_product"], "shows_damage": row["shows_damage"]})

def pick(samples):
    # balanced set: per product, damaged and good photos; 20% get the wrong product name
    rnd = random.Random(0)
    rows = []
    for category in PRODUCTS:
        mine = [s for s in samples if s["category"]["label"] == category]
        for damaged in [True, False]:
            group = [s for s in mine if (s["defect"]["label"] != "good") == damaged]
            for s in rnd.sample(group, min(PER_CLASS, len(group))):
                wrong = rnd.random() < 0.2
                product = rnd.choice([c for c in PRODUCTS if c != category]) if wrong else category
                rows.append({"file": s["filepath"], "category": category, "defect": s["defect"]["label"],
                             "product": PRODUCTS[product], "shows_product": not wrong, "shows_damage": damaged})
    rnd.shuffle(rows)
    return rows[TEST_SIZE:], rows[:TEST_SIZE]

if __name__ == "__main__":
    samples = json.loads(Path(hf_hub_download(REPO, "samples.json", repo_type="dataset")).read_text())["samples"]
    train, test = pick(samples)
    print(f"train {len(train)}  test {len(test)}  damaged in test {sum(r['shows_damage'] for r in test)}"
          f"  wrong product in test {sum(not r['shows_product'] for r in test)}")

    # download only the photos we use, shrink them to 448px JPEGs
    raw = snapshot_download(REPO, repo_type="dataset", allow_patterns=[r["file"] for r in train + test])
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for r in train + test:
        r["image"] = f"img/{r['file'].replace('/', '_').replace('.png', '.jpg')}"
        Image.open(Path(raw) / r["file"]).convert("RGB").resize((448, 448)).save(OUT / r["image"], quality=90)
    for name, rows in [("train", train), ("test", test)]:
        (OUT / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {OUT}")
