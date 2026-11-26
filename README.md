# ReturnLoop Agent

An AI agent that investigates e-commerce return and refund claims. It looks up the claim,
the customer's history, the carrier status and the merchant policy, then decides
**approve / reject / review** and writes a reply to the customer.

The tools run behind an **MCP server**. Policy rules come from **hybrid search** (BM25 + embeddings,
reciprocal rank fusion) over 5 policy documents, and every verdict must cite the policy sections
it used. Citations the agent never retrieved are flagged as ungrounded.
See `CLAUDE.md` for the coding rules and the full plan.

```
py -m pip install -r requirements.txt
copy .env.example .env
py data\make_data.py
py search.py
py agent.py
```

| File | What |
|---|---|
| `data/make_data.py` | builds `returnloop.db`: customers, orders, deliveries, claims (with expected decisions) |
| `data/policies/*.md` | the merchant's return policy, 5 docs, 19 sections |
| `data/policy_questions.json` | 20 labelled questions for the search eval |
| `search.py` | chunking, BM25 + dense + RRF hybrid search, recall@3 / MRR eval |
| `mcp_server.py` | MCP server: get_claim, get_customer, get_delivery, search_policy |
| `agent.py` | MCP client + tool-calling loop, cited verdict, accuracy and grounding vs expected |
| `llm_config.py` | Gemini client through the OpenAI library |

## Results

Policy search on 20 labelled questions (`py search.py`):

| | recall@3 | MRR |
|---|---|---|
| BM25 | 0.90 | 0.82 |
| Dense (potion-base-8M) | 0.95 | 0.91 |
| **Hybrid (RRF)** | **1.00** | 0.83 |

Before fixing tokenisation (numbers like 20,000 split, no stemming) and adding doc titles to chunks,
hybrid recall@3 was 0.81, below dense alone.
