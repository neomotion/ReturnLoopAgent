import random

from vision_eval import apply_threshold, parse, pick_threshold, score

def test_parse_reads_json_inside_markdown_and_rejects_prose():
    assert parse('```json\n{"shows_product": true, "shows_damage": false}\n```') == {"shows_product": True, "shows_damage": False}
    assert parse("it looks damaged") == {}

def test_threshold_is_the_best_one_with_high_precision():
    rnd = random.Random(0)
    rows = [{"shows_damage": rnd.random() < 0.5, "shows_product": True} for _ in range(200)]
    probs = [rnd.uniform(0.3, 0.9) if r["shows_damage"] else rnd.uniform(0.0, 0.5) for r in rows]
    preds = [{"shows_product": True} for _ in rows]
    t = pick_threshold(rows, preds, probs)
    at = lambda x: score(rows, apply_threshold(preds, probs, x))
    assert at(t)["damage_precision"] >= 0.95
    best = max(at(x / 100)["damage_recall"] for x in range(5, 96) if at(x / 100)["damage_precision"] >= 0.95)
    assert at(t)["damage_recall"] == best
