import json
import sys

from pydantic import BaseModel, Field

from agent import ROOT
from llm_config import client, MODEL

LABELS = ROOT / "data" / "reply_labels.json"
CRITERIA = ["matches_decision", "no_internal_details", "clear_next_step"]

JUDGE = """You grade a customer-service reply about a return claim. Answer each check with true or false.
matches_decision: the reply says the same as the decision. No refund is promised for reject or review.
no_internal_details: the reply does not reveal risk signals (return rate, account age, fraud suspicion,
  serial returner, reused photo) or internal policy section ids.
clear_next_step: the reply tells the customer what happens next."""

class ReplyCheck(BaseModel):
    reasoning: str = Field(description="one or two sentences, written before the checks")
    matches_decision: bool
    no_internal_details: bool
    clear_next_step: bool

def judge(case):
    res = client.chat.completions.parse(model=MODEL, temperature=0, response_format=ReplyCheck, messages=[
        {"role": "system", "content": JUDGE},
        {"role": "user", "content": f"Decision: {case['decision']}\nReply: {case['reply']}"}])
    return res.choices[0].message.parsed.model_dump()

def kappa(a, b):
    # Cohen's kappa for two lists of true/false: agreement corrected for agreement by chance
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    chance = (sum(a) / n) * (sum(b) / n) + (1 - sum(a) / n) * (1 - sum(b) / n)
    return 1.0 if chance == 1 else round((observed - chance) / (1 - chance), 3)

def label():
    # you grade the replies from the last evaluate.py run; the judge is then measured against you
    cases = json.loads((ROOT / "results" / "eval.json").read_text())["cases"]
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    for c in cases:
        if c["id"] in labels:
            continue
        print(f"\n{c['id']}  decision: {c['decision']}\nreply: {c['reply']}")
        labels[c["id"]] = {"decision": c["decision"], "reply": c["reply"]} | {
            k: input(f"  {k}? (y/n) ").strip().lower() == "y" for k in CRITERIA}
        LABELS.write_text(json.dumps(labels, indent=1))   # saved after each one, so you can stop any time

def calibrate():
    labels = json.loads(LABELS.read_text())
    judged = {cid: judge(case) for cid, case in labels.items()}
    for k in CRITERIA:
        human, model = [labels[c][k] for c in labels], [judged[c][k] for c in labels]
        disagree = [c for c in labels if labels[c][k] != judged[c][k]]
        print(f"{k:20} agreement {sum(h == m for h, m in zip(human, model))}/{len(human)}"
              f"  kappa {kappa(human, model)}  judge pass rate {sum(model)}/{len(model)}  disagree: {disagree}")
    return judged

# ---- py judge.py label  (you grade replies)   |   py judge.py  (judge vs you: agreement + kappa) ----
if __name__ == "__main__":
    label() if sys.argv[1:] == ["label"] else calibrate()
