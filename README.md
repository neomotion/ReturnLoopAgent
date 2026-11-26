# ReturnLoop Agent

An AI agent that investigates e-commerce return and refund claims. It looks up the claim,
the customer's history, the carrier status and the merchant policy, then decides
**approve / reject / review** and writes a reply to the customer.

Phase 1: one tool-calling agent, 8 labelled claims, accuracy printed at the end.
See `CLAUDE.md` for the coding rules and the full plan.

```
py -m pip install -r requirements.txt
copy .env.example .env
py data\make_data.py
py agent.py
```

| File | What |
|---|---|
| `data/make_data.py` | builds `returnloop.db`: customers, orders, deliveries, claims (with expected decisions) |
| `data/policy.md` | the merchant's return policy |
| `tools.py` | the 4 tools the agent can call + their schemas |
| `agent.py` | tool-calling loop, structured verdict, accuracy vs expected |
| `llm_config.py` | Gemini client through the OpenAI library |
