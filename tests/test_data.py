import json
import sqlite3
from collections import Counter

def test_golden_set_covers_every_decision_and_rule():
    rows = sqlite3.connect("data/returnloop.db").execute("select id, expected, rules from claims").fetchall()
    gold = [r for r in rows if r[0].startswith("g")]
    assert len(gold) == 200
    assert set(Counter(r[1] for r in gold)) == {"approve", "reject", "review"}
    rules = {rule for r in gold for rule in json.loads(r[2])}
    assert {"risk#high-value-refunds", "risk#serial-returners", "risk#new-accounts",
            "risk#instructions-inside-claim-text", "delivery#lost-by-carrier"} <= rules
