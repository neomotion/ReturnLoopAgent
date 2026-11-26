import asyncio
import json
import random
import sqlite3
import sys
import time
from collections import Counter

from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client

from agent import ROOT, SERVER, to_openai
from graph import builder, ctx

# the gate: below any of these, the run fails (exit code 1), so a bad prompt or model change can't ship
GATE = {"accuracy": 0.85, "rule_recall": 0.90, "ungrounded": 0}

def sample(n, seed=0):
    # the 9 hand claims + n golden claims, with the golden set's share of approve / reject / review
    rows = sqlite3.connect(ROOT / "data" / "returnloop.db").execute("select id, expected, rules from claims").fetchall()
    rows = [{"id": r[0], "expected": r[1], "rules": json.loads(r[2])} for r in rows]
    gold = [r for r in rows if r["id"].startswith("g")]
    rnd, picked = random.Random(seed), []
    for decision in ["approve", "reject", "review"]:
        group = [r for r in gold if r["expected"] == decision]
        picked += rnd.sample(group, round(n * len(group) / len(gold)))
    return [r for r in rows if r["id"].startswith("k")] + picked

async def run_case(graph, case):
    config = {"configurable": {"thread_id": case["id"]}, "recursion_limit": 20}
    start = time.time()
    await graph.ainvoke({"claim_id": case["id"]}, config)
    state = (await graph.aget_state(config)).values   # finished, or paused at human_review: the verdict is there
    v = state["verdict"]
    return case | {"decision": v["decision"], "citations": v["citations"], "seen": state["seen"], "reply": v["reply"],
                   "seconds": round(time.time() - start, 1)}

def report(cases):
    # decisions, plus the trajectory: did the policy agent retrieve every rule the case needs?
    required = [(rule, rule in c["seen"]) for c in cases for rule in c["rules"]]
    return {
        "cases": len(cases),
        "accuracy": round(sum(c["decision"] == c["expected"] for c in cases) / len(cases), 3),
        "confusion (expected -> got)": {f"{e} -> {g}": n for (e, g), n in sorted(Counter(
            (c["expected"], c["decision"]) for c in cases).items())},
        "rule_recall": round(sum(hit for _, hit in required) / len(required), 3),
        "rules_missed": dict(Counter(rule for rule, hit in required if not hit).most_common()),
        "ungrounded": sum(len([x for x in c["citations"] if x not in c["seen"]]) for c in cases),
        "seconds_per_case": round(sum(c["seconds"] for c in cases) / len(cases), 1),
    }

def gate(summary):
    fails = [k for k, limit in GATE.items()
             if (summary[k] > limit if k == "ungrounded" else summary[k] < limit)]
    return fails

async def main(n):
    cases = sample(n)
    async with Client(SERVER) as mcp:
        ctx["mcp"] = mcp
        ctx["tools"] = [to_openai(t) for t in (await mcp.list_tools()).tools]
        graph = builder.compile(checkpointer=InMemorySaver())
        done = []
        for case in cases:
            done.append(await run_case(graph, case))
            c = done[-1]
            missed = [r for r in c["rules"] if r not in c["seen"]]
            print(f"{c['id']}: {c['decision']} (expected {c['expected']})" + (f"  MISSED RULES {missed}" if missed else ""))
    summary = report(done)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "eval.json").write_text(json.dumps({"summary": summary, "cases": done}, indent=1))
    print(json.dumps(summary, indent=1))
    fails = gate(summary)
    print("GATE PASSED" if not fails else f"GATE FAILED: {fails}")
    return fails

# ---- py evaluate.py [n golden cases, default 40] ----
if __name__ == "__main__":
    sys.exit(1 if asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)) else 0)
