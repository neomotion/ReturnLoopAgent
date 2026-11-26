# ReturnLoop Agent

A multi-agent system that investigates e-commerce return / refund claims, decides
approve / reject / review, asks a human before risky refunds, and replies to the customer.
Built to learn production agent engineering and to be a resume project.

- Local folder: `D:\sahil\ReturnLoopAgent` (Windows, run with `py`)
- GitHub: https://github.com/neomotion/ReturnLoopAgent (push after every update)
- LLM: Gemini through the OpenAI library (`llm_config.py`, key in `.env`)
- Data: synthetic, or public datasets (MVTec AD for the vision model). Never use Pinch data or code.

## Coding rules (read before writing any code)

1. **Minimal.** Write the least code that works today. No feature, option, or layer until it is needed.
2. **Simple.** Someone should understand a file by reading it once, top to bottom.

How that looks in practice (taken from `tool_calling_loop.py`, `speed_test_json.py`, `practice.py`):

- **Flat scripts.** Order: imports → setup → data → functions → run. Code runs top to bottom;
  put the run part under `if __name__ == "__main__":` when another file imports this one.
- **Section comments** like `# ---- LLM setup ----` and short lowercase comments above a block
  (`# loop`, `# convert to text`). Say what the block does. No long docstrings.
- **Plain data.** Dicts and lists. No classes except Pydantic models (LLM output) and the
  LangGraph `TypedDict` state. No inheritance, factories, managers, or config systems.
- **Small functions** that do one thing and return a value (often a dict).
- **LLM calls** look the same everywhere: `client.chat.completions.create(model=MODEL, temperature=0, messages=...)`.
  Structured output with `client.chat.completions.parse(..., response_format=SomeModel)`.
- **Tools** live in `mcp_server.py` as `@mcp.tool()` functions: type hints + a one-line docstring
  are the schema the model sees. A tool returns a dict, and `{"error": "..."}` instead of raising.
  (Outside MCP: a function + a JSON-schema entry in `TOOLS` + a `FUNCTIONS = {"name": fn}` map.)
- **Loops have a hard cap**: `for step in range(MAX_STEPS):` and break on the final answer.
- **Print what happens** (tool calls, matches, verdicts) so a run explains itself. No logging library.
- **Paths** with `pathlib`, relative to `__file__`.
- **No defensive code for things that haven't happened.** No try/except, retries, or validation
  until a real failure shows up; then add the smallest fix.
- **Short plain names**: `data`, `reply`, `args`, `f`, `rows`.
- **One file per idea**, ideally under ~150 lines. A new library only when it removes real code.

## Plan

| Phase | What | Learn |
|---|---|---|
| 1 ✅ | Single agent: claim, customer, delivery, policy tools; structured verdict; score vs expected | tool loop, structured output |
| 2 ✅ | Tools as an MCP server; policy RAG (hybrid search + citations) over policy docs | MCP, agentic RAG |
| 3 ✅ | LangGraph graph of specialist agents (evidence, policy, decide), human approval via `interrupt()`, SQLite checkpointer (Postgres when deployed) | multi-agent, HITL, durability |
| 4 ✅ | Vision step (does the photo show the damage?) + reused-photo check with a perceptual hash | multimodal |
| 5 ✅ | Policy as code (`policy_rules.py`) labels a 200-case golden set; `evaluate.py` (accuracy, confusion, required-rule recall, grounding, gate); reply judge with Cohen's κ; pytest + GitHub Actions | evals |
| 6 | Prompt-injection red-team suite, tool permissions | security |
| 7 | Langfuse tracing, cost/latency per case, model routing, caching | observability, cost |
| 4b | Local vision model: Qwen3-VL-2B fine-tuned with QLoRA on real defect photos (MVTec AD) in Colab; base vs tuned vs Gemini on the same 200-photo test set | fine-tuning, VLMs |
| 8 | Merchant memory from reviewer overrides | memory |
| 9 | FastAPI streaming + UI, Docker, deploy | serving |

## Run

```
py -m pip install -r requirements.txt
copy .env.example .env      (then put your key in .env)
py data\make_data.py        (creates data\returnloop.db and data\photos)
py photos.py                (perceptual hash check: md5 vs dHash distances)
py search.py                (policy search eval: recall@3 / MRR for bm25, dense, hybrid)
py agent.py                 (single agent: starts the MCP server, runs every claim, prints verdict vs expected)
py graph.py                 (multi-agent graph; asks you approve/reject for review cases; resumes after a restart)
py evaluate.py [n]          (graph on the 9 hand claims + n golden ones; exits 1 if the gate fails)
py judge.py label           (you grade the replies from the last evaluate run)
py judge.py                 (LLM judge vs your grades: agreement + Cohen's kappa)
pytest tests                (fake model, no API key: data, search, photos, eval gate, judge, vision eval)
py vision_data.py           (downloads ~1 GB once: 590 train / 200 test real defect photos into data\vision)
py vision_eval.py           (Gemini on the 200 test photos: damage precision/recall/F1, product accuracy)
train_vision.ipynb          (Colab T4: base Qwen3-VL-2B eval, QLoRA fine-tune, tuned eval)
```

## Testing

Test every stage before pushing, and check it does what we expect, not just that it runs:
- `pytest tests` must pass (GitHub Actions runs it on every push). New behaviour gets a test in `tests/`.
- `tests/fake_llm.py` is a fake Gemini that behaves like a good agent; planted mistakes (FAKE_WRONG,
  FAKE_UNGROUNDED, FAKE_SKIP_RISK) prove the checks catch problems. Gemini is not reachable from
  Claude's workspace, so real-model runs happen on Sahil's laptop or the manual `real-model eval` workflow.
- Break the thing a test guards once, to see the test go red.
- Report measured numbers.

## Git

Small commits, one change each, message says what changed (`add delivery tool`). Push after every update.
Commits are authored by `neomotion`. No Claude / AI attribution anywhere (no Co-Authored-By lines).
