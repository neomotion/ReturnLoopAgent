# ReturnLoop Agent

A multi-agent system that investigates e-commerce return / refund claims, decides
approve / reject / review, asks a human before risky refunds, and replies to the customer.
Built to learn production agent engineering and to be a resume project.

- Local folder: `D:\sahil\ReturnLoopAgent` (Windows, run with `py`)
- GitHub: https://github.com/neomotion/ReturnLoopAgent (push after every update)
- LLM: Gemini through the OpenAI library (`llm_config.py`, key in `.env`)
- Data: synthetic only. Never use Pinch data or code.

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
- **Tools** = a Python function + a JSON-schema entry in `TOOLS` + a `FUNCTIONS = {"name": fn}` map.
  Tool results go back as `json.dumps(result)`. A tool returns `{"error": "..."}` instead of raising.
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
| 1 | Single agent: claim, customer, delivery, policy tools; structured verdict; score vs expected | tool loop, structured output |
| 2 | Tools as an MCP server; policy RAG (hybrid search + citations) over policy docs | MCP, agentic RAG |
| 3 | LangGraph supervisor + specialist agents, human approval via `interrupt()`, Postgres checkpointer | multi-agent, HITL, durability |
| 4 | Vision agent (damage photos, reused-photo check) | multimodal |
| 5 | Golden set (200 cases), LLM judge with κ, trajectory checks, CI gate | evals |
| 6 | Prompt-injection red-team suite, tool permissions | security |
| 7 | Langfuse tracing, cost/latency per case, model routing, caching | observability, cost |
| 8 | Merchant memory from reviewer overrides; LoRA-distilled triage model | memory, fine-tuning |
| 9 | FastAPI streaming + UI, Docker, deploy | serving |

## Run

```
py -m pip install -r requirements.txt
copy .env.example .env      (then put your key in .env)
py data\make_data.py        (creates data\returnloop.db)
py agent.py                 (runs every claim, prints verdict vs expected)
```

## Git

Small commits, one change each, message says what changed (`add delivery tool`). Push after every update.
