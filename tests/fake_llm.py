# a fake Gemini that behaves like a well-run agent, so tests can check the harness without an API key.
# planted mistakes: FAKE_WRONG="k4,g005" flips those decisions, FAKE_UNGROUNDED="k6" adds a made-up citation,
# FAKE_SKIP_RISK=1 makes the policy agent forget the risk rules (the single-agent k6 failure).
import json
import os
import types

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

import llm_config
from policy_rules import expected

RISK_QUERIES = ["high value refunds", "serial returners", "new accounts", "instructions inside claim text",
                "photo does not match the claim", "reused photo"]
TYPE_QUERIES = {"damaged": "damaged item", "wrong_item": "wrong item", "changed_mind": "change of mind",
                "not_received": "parcel not received lost in transit delivered with proof"}

def reply(text=None, calls=()):
    tool_calls = [ChatCompletionMessageToolCall(id=f"t{i}", type="function", function=Function(name=n, arguments=json.dumps(a)))
                  for i, (n, a) in enumerate(calls)] or None
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls))])

def parsed(obj):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=obj))])

def field(task, name):
    # "Claim: {...}\nEvidence: {...}" -> the JSON after "Name: "
    return json.loads(task.split(f"{name}: ", 1)[1].split("\n")[0])

def create(**kw):
    messages, system = kw["messages"], kw["messages"][0]["content"]
    first = not any(m["role"] == "assistant" for m in messages)
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if "evidence agent" in system:
        claim = json.loads(messages[1]["content"])
        if first:
            return reply(calls=[("get_customer", {"customer_id": claim["customer_id"]}),
                                ("get_delivery", {"order_id": claim["order_id"]})])
        return reply(json.dumps({"customer": results[0], "delivery": results[1]}))
    if "policy agent" in system:
        claim = field(messages[1]["content"], "Claim")
        if first:
            queries = [TYPE_QUERIES[claim["type"]], claim["category"]]
            queries += [] if os.environ.get("FAKE_SKIP_RISK") else RISK_QUERIES
            return reply(calls=[("search_policy", {"query": q}) for q in queries])
        ids = sorted({r["id"] for res in results for r in res["results"]})
        return reply("Relevant sections: " + json.dumps(ids))
    raise AssertionError(f"unexpected create() call: {system[:60]}")

def parse(**kw):
    model, task = kw["response_format"], kw["messages"][1]["content"]
    if model.__name__ == "PhotoCheck":
        assert task[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return parsed(model(shows_product=True, shows_damage="mug" not in task[0]["text"], note="fake"))
    if model.__name__ == "Verdict":
        claim, evidence, photo = field(task, "Claim"), field(task, "Evidence"), field(task, "Photo")
        seen = json.loads(task.split("Relevant sections: ", 1)[1].split("\n")[0])
        facts = {"shows_damage": photo["shows_damage"], "reused": bool(photo["reused_from"])} if photo.get("has_photo") else {}
        decision, rules = expected(claim, evidence["customer"], evidence["delivery"], facts)
        if claim["id"] in os.environ.get("FAKE_WRONG", "").split(","):
            decision = {"approve": "reject", "reject": "approve", "review": "approve"}[decision]
        citations = [r for r in rules if r in seen]
        if claim["id"] in os.environ.get("FAKE_UNGROUNDED", "").split(","):
            citations.append("risk#made-up-rule")
        return parsed(model(reasoning="fake", decision=decision, citations=citations, reply="fake reply"))
    if model.__name__ == "ReplyCheck":
        decision, text = task.split("\n", 1)[0].split(": ")[1], task.split("Reply: ", 1)[1].lower()
        return parsed(model(reasoning="fake", matches_decision=decision == "approve" or "refund" not in text,
                            no_internal_details="return rate" not in text and "#" not in text,
                            clear_next_step="will" in text))
    raise AssertionError(f"unexpected parse() call: {model.__name__}")

def install():
    llm_config.client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create, parse=parse))
