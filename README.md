# Agentic AI Capstone: E-Commerce Customer Support Assistant

This project implements a phase-structured Agentic AI system using **LangGraph + ChromaDB RAG + evaluation pipeline**, aligned with the capstone workflow.

## Domain and Scope
- **Domain:** E-Commerce Customer Support Assistant
- **User:** Online shoppers (orders, returns, refunds, shipping, product-policy queries)
- **Success criteria:** Grounded answers, correct retrieval and routing, thread memory continuity, tool use, safe handling of out-of-scope/injection prompts

## Deliverables
1. `agent.py` — complete capstone agent pipeline (Phases 1–6)
2. `capstone_streamlit.py` — Streamlit deployment (Phase 7)
3. `day13_capstone_completed.ipynb` — completed notebook code
4. `README.md` — structured submission guide

## Project Structure
| File | Purpose |
|---|---|
| `agent.py` | Knowledge base, retrieval tests, `CapstoneState`, nodes, graph, tests, evaluation |
| `capstone_streamlit.py` | Chat UI with cached agent, session memory, and thread reset |
| `day13_capstone_completed.ipynb` | Notebook flow mirroring all phases |
| `requirements.txt` | Python dependencies |

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set API key (optional but recommended for LLM quality):
   ```bash
   set GROQ_API_KEY=your_key_here
   ```

If `GROQ_API_KEY` is missing or invalid, the project automatically falls back to a local deterministic LLM stub so the pipeline still runs.
If `sentence-transformers`/`torch` cannot be loaded in your environment, it falls back to a local hash embedder (the code still prefers `SentenceTransformer('all-MiniLM-L6-v2')` first).

## Run Phases 1–6 (script)
```bash
python agent.py
```

What this runs:
1. Builds 10-document KB using `SentenceTransformer('all-MiniLM-L6-v2')` + in-memory ChromaDB
2. Runs retrieval gate tests
3. Runs node-level tests
4. Compiles graph with `MemorySaver`
5. Executes phase-5 query suite (incl. red-team tests)
6. Executes phase-6 evaluation (RAGAS if available, else LLM fallback)

## Run Streamlit (Phase 7)
```bash
streamlit run capstone_streamlit.py
```

## LangGraph Flow
`memory -> router -> retrieve/tool/skip -> answer -> eval -> save -> END`

Decision functions:
- `route_decision()` for retrieve/tool/skip routing
- `eval_decision()` for retry-vs-save (faithfulness threshold + retry cap)

## Testing and Evaluation Coverage
- **Phase 5:** 10+ test queries including:
  - normal retrieval queries
  - tool query (datetime/calculator)
  - memory query (name recall by `thread_id`)
  - 2 red-team tests (out-of-scope + prompt injection)
  - explicit PASS/FAIL judgement per test
  - dedicated 3-turn memory sequence test (`run_memory_sequence_test`)
- **Phase 6:** 5 QA ground-truth pairs evaluated with:
  - `faithfulness`
  - `answer_relevancy`
  - `context_precision`

## Notes
- Tool node is fail-safe and always returns a string.
- Answer node enforces strict grounding and uncertainty response when context is insufficient.
- Node writes are constrained to fields declared in `CapstoneState`.

## Helper Document Alignment
This implementation follows the helper document structure:
1. Phase-0 framing (domain, user, success criteria) completed before coding.
2. Six mandatory capabilities covered: LangGraph, ChromaDB RAG, MemorySaver/thread_id, eval retry gate, tool use, Streamlit deployment.
3. 8-part capstone flow mapped in `agent.py` and notebook.
4. Retrieval tested before graph assembly, with a hard quality gate.

## Warning Compliance (Helper PDF)
The helper document warnings are explicitly handled:
1. Retrieval is verified before node tests/graph assembly (hard gate).
2. `CapstoneState` is defined before node functions and includes required base fields.
3. `tool_node` is fail-safe and always returns strings instead of raising.
4. Graph path includes `save -> END`, with compile verification.
5. Phase-5 PASS/FAIL uses rule checks (not answer-length heuristics).
6. Streamlit file is UTF-8 readable on Windows (`helper_warnings_status` check).
7. No placeholder TODO text remains in submission artifacts.
