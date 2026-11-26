import asyncio
import json

import fake_llm
import evaluate

fake_llm.install()

def run(monkeypatch, **env):
    # 9 hand claims + 10 golden ones through the real graph, MCP server and search, with the fake model
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return asyncio.run(evaluate.main(10))

def summary():
    return json.loads((evaluate.ROOT / "results" / "eval.json").read_text())["summary"]

def test_good_agent_passes_the_gate(monkeypatch):
    fails = run(monkeypatch)
    assert fails == [] and summary()["accuracy"] == 1.0 and summary()["rule_recall"] == 1.0

def test_wrong_decisions_fail_on_accuracy(monkeypatch):
    fails = run(monkeypatch, FAKE_WRONG="k1,k2,k4")
    assert fails == ["accuracy"] and summary()["accuracy"] < 0.85

def test_made_up_citation_fails_on_grounding(monkeypatch):
    fails = run(monkeypatch, FAKE_UNGROUNDED="k6")
    assert fails == ["ungrounded"] and summary()["ungrounded"] == 1

def test_skipped_risk_rules_fail_even_when_decisions_are_right(monkeypatch):
    fails = run(monkeypatch, FAKE_SKIP_RISK="1")
    s = summary()
    assert fails == ["rule_recall"] and s["accuracy"] == 1.0 and "risk#high-value-refunds" in s["rules_missed"]
