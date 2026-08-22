# Project Brief: Regulatory Adjudicator Agent (FDA vs. EMA)

## 1. Context — who is building this and why

I'm a 2nd-year Computer Engineering student (going into 3rd year) building a portfolio project for CV/internship applications to bigger tech companies. My existing background: C systems programming (async epoll web server), embedded/RTOS work (stratospheric UAV autonomy stack), some crypto on microcontrollers, one ML pipeline (SMOTE-based classifier), and full-stack work (React/Node, OAuth2, REST APIs).

I do **not** have prior experience with agentic LLM systems, RAG, or vector databases. I have used Claude/ChatGPT as end-user tools but have not built a tool-calling agent from scratch. Explanations should assume I understand general programming (Python is fine, I've used it) but should **explain LLM-specific concepts** (tool calling, embeddings, ReAct loop, retrieval) as they come up rather than assuming I already know them — I want to actually understand this project well enough to defend every design decision in a technical interview, not just have it work.

## 2. What this project needs to prove (this drives every design decision)

The point of this project is **not** "an AI that answers pharma questions." The point is to demonstrate:

1. **Dynamic tool selection** — the agent decides which sources to query based on the question, rather than following a hardcoded call sequence. This must be visibly demonstrable (e.g., a US-only question doesn't trigger an EU search).
2. **Self-correction (verify-and-retry)** — a second pass checks every factual claim in the draft answer against the actual retrieved source text, and triggers a correction if a claim isn't supported. This is the single most important artifact for interviews — I need at least one clean, logged/screenshotted example of the verifier catching and fixing a bad claim.
3. **Grounded citations** — every claim in the final output traces back to a specific source chunk, not model memory.
4. **Measurable correctness** — a small automated eval suite that checks tool-call trajectory, conflict classification accuracy, and citation validity, not just "it looks right when I tried it."

Everything else (data volume, UI polish, number of drugs) is secondary to these four things working cleanly.

## 3. Scope — what's in and out for this build

**Timeframe:** weekend-plus (aiming for 2-3 focused days, not a multi-week build).

**In scope:**
- 4 drugs with well-documented, real FDA/EMA divergence
- 2 cross-cutting regulatory guidance documents (1 FDA, 1 EMA)
- Single-turn Q&A (ask a question, get a report — no multi-turn memory/session state)
- Dynamic tool-calling ReAct loop, hand-written (no LangChain/CrewAI — the point is to understand and be able to explain the mechanics, not to hide them behind a framework)
- Verifier module as a distinct step, not folded into the main generation call
- 10-12 hand-written eval cases with an automated runner

**Explicitly out of scope for v1** (note these as "future work" in the README, don't build them now):
- Multi-turn session memory / state ledger
- Automated PDF-to-markdown ingestion pipeline (manual conversion is fine for 4 drugs)
- More than 4 drugs
- Any UI beyond a CLI or a minimal single-page demo
- Metadata-filtered vector search (plain semantic search is enough at this scale)
- Observability dashboards (nice later, not now)

## 4. The 4 drugs (real, documented divergence — not invented for the demo)

| Drug | Indication | FDA posture | EMA posture |
|---|---|---|---|
| Aducanumab (Aduhelm) | Alzheimer's | Accelerated Approval based on amyloid-beta reduction (surrogate endpoint) | Refused marketing authorization — clinical benefit not established |
| Lecanemab (Leqembi) | Early Alzheimer's | Broad approval, standard monitoring | Approved with restriction — recommended mainly for ApoE ε4 non-carriers/heterozygotes due to ARIA (brain swelling/bleeding) risk |
| Bevacizumab (Avastin) | Metastatic breast cancer | Indication revoked (2011) — no overall survival benefit shown | Indication retained, in combination with paclitaxel |
| Olaparib (Lynparza) | Prostate cancer | Restricted to BRCA-mutated mCRPC | Approved for broader HRR gene-mutated population |

Include one "harmonized" or "insufficient data" test case in the eval set too, so the agent is demonstrably not just pattern-matching "always find a conflict."

## 5. Data sources (confirmed public, no registration/approval wait)

- **FDA drug labels:** `openFDA` REST API (`api.fda.gov/drug/label.json`) — free, no API key required for light use. Returns label text pre-split into sections (indications, dosage, contraindications, clinical studies).
- **FDA guidance docs:** direct PDF download from fda.gov (e.g., Accelerated Approval Program guidance).
- **EMA data:** EMA's per-product pages link directly to the SmPC PDF; EMA also publishes site data in JSON format for automated use. For 4 drugs, manually downloading each SmPC PDF is fine — don't over-engineer ingestion.
- **EMA guidance docs:** direct PDF from ema.europa.eu (e.g., conditional marketing authorisation guideline).

**PDF → Markdown conversion:** use an LLM with a strict prompt that forbids inventing content and requires a fixed section structure mirroring the source (e.g., "Therapeutic Indications," "Posology," "Contraindications" for SmPCs). This is a conversion task, not a generation task — the prompt must make that constraint explicit, and you should spot-check the output against the source PDF before trusting it.

## 6. Architecture

```
regulatory-adjudicator-mini/
├── data/
│   ├── raw/                    # source PDFs/JSON, gitignored
│   ├── processed/
│   │   └── drugs_metadata.csv  # 4 rows: drug_id, brand_name, generic_name,
│   │                            #   drug_class, indication, fda_status,
│   │                            #   fda_pathway, ema_status, ema_pathway
│   └── docs/
│       ├── drugs/               # {drug}_fda.md, {drug}_ema.md — 8 files
│       └── guidance/            # 2 files: fda_accelerated_approval.md,
│                                 #          ema_conditional_authorisation.md
├── evals/
│   ├── evalset.json             # 10-12 test cases
│   └── run_evals.py
├── src/
│   ├── ingest.py                 # fetch/convert both sources, one file, clearly
│   │                              # separated functions (fetch_fda, fetch_ema,
│   │                              # fetch_guidance)
│   ├── retrieval.py               # chunk + embed, single ChromaDB collection,
│   │                              # search_fda(query), search_ema(query)
│   ├── tools.py                   # tool JSON schemas + execution wrappers:
│   │                              # get_drug_metadata, search_fda, search_ema
│   ├── agent.py                   # ReAct loop: LLM decides tool calls -> execute
│   │                              # -> draft -> verify -> retry (max 2) -> return
│   └── verify.py                  # separate LLM call: checks each cited claim
│                              #   in the draft against the actual retrieved
│                              #   chunk text, returns pass/fail per claim
├── demo.py                        # CLI: ask a question, print the full trace
│                                   #   (thought -> tool call -> observation ->
│                                   #   draft -> verify -> final) and the report
├── README.md
└── requirements.txt
```

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vector DB | ChromaDB, local persistent | Zero setup, `pip install chromadb`, no external service |
| Agent loop | Hand-written Python `while` loop + Anthropic SDK tool calling | Proves you understand tool-calling at the protocol level — no LangChain/CrewAI, since frameworks hide exactly the mechanics you need to be able to explain in an interview |
| Data validation | Pydantic | Tool input/output schemas, eval case schema — catches structural errors early and signals disciplined Python |
| PDF handling | Manual extraction or `pymupdf`, plus one LLM conversion pass per doc | At 4 drugs, a full OCR/ingestion pipeline is overkill |
| LLM | Claude (Sonnet-tier model via Anthropic SDK) | Tool-calling support, reasonable cost at this scale |
| Eval runner | Plain Python + `pytest`, custom assertions against `evalset.json` | Transparent, no black-box eval framework needed to explain what's being checked |

## 8. Core loop, concretely

```
User question
    │
    ▼
Agent (LLM) reads question + tool descriptions
    │
    ├─ decides which tool(s) to call (get_drug_metadata / search_fda / search_ema)
    │  — this decision must vary by question type; test this explicitly
    ▼
Execute tool(s), collect results with chunk IDs
    │
    ▼
Draft agent: writes report citing specific chunk IDs per claim
    │
    ▼
Verifier: for each cited claim, pull the exact chunk text, check whether
it actually supports the claim → {claim, chunk_id, supported: bool, reason}
    │
    ├─ all claims supported → return report
    │
    └─ some claim fails → feed specific failure back to draft agent,
       regenerate just that section, re-verify (cap at 2 retries,
       then mark the claim "insufficient evidence" rather than guessing)
```

**System prompt guardrails to bake in (adapt language, keep the constraints):**
- Never answer from model memory — every factual claim must come from a tool result.
- When evaluating a drug or trial-design question, check both FDA and EMA sources before concluding — unless the question is explicitly single-jurisdiction.
- Explicitly label the outcome as harmonized, divergent, or insufficient evidence.
- Every assertion must reference the specific chunk it came from.

## 9. Eval suite — what "done" looks like

10-12 cases in `evalset.json`, each with:
```json
{
  "test_id": "eval_003_aducanumab_jurisdiction",
  "query": "Is Aducanumab authorized for prescription in both the US and EU markets?",
  "expected_posture": "DIVERGENT",
  "expected_tool_trajectory": ["get_drug_metadata", "search_fda", "search_ema"],
  "required_fact_assertions": ["FDA accelerated approval", "EMA refused", "surrogate endpoint"],
  "forbidden_hallucinations": ["EMA approved Aducanumab", "available in all European pharmacies"]
}
```
Include: 2-3 cases per drug, 1 harmonized/no-conflict case, 1 out-of-scope or insufficient-data case (tests it doesn't fabricate an answer when it shouldn't have one). `run_evals.py` checks trajectory match, required facts present, forbidden phrases absent — print a pass/fail table, don't aim for 100%; an honest "9/12 pass, here's why the other 3 fail" is a better README than a suspiciously perfect suite.

## 10. Build order (day by day)

**Day 1 — data + retrieval only, no LLM agent yet**
1. Pull 4 drugs' FDA labels (openFDA) + EMA SmPCs (manual PDF download), convert to markdown.
2. Pull + convert 2 guidance docs.
3. Chunk everything, load into ChromaDB.
4. Write and directly test `search_fda()`, `search_ema()`, `get_drug_metadata()` as plain Python functions — sanity-check output by hand before any LLM touches them.

**Day 2 — the agent loop**
5. Define tool schemas, write the ReAct loop. Test on one question, print every intermediate step so tool-selection is visibly happening, not scripted.
6. Write `verify.py` — separate call, checks draft claims against retrieved chunks.
7. Wire in the retry path (cap 2 attempts).

**Day 3 (or spillover)**
8. Write the 10-12 eval cases by hand (this also double-checks your source data is correct).
9. Write and run `run_evals.py`, fix obvious failures, document what still fails and why.
10. Capture one clean example of the verifier catching a bad claim — this is your best interview artifact, save the trace/logs or a screenshot.

## 11. What NOT to claim (for README, CV bullets, interviews)

- Not "zero hallucinations" — the verifier reduces unsupported claims, doesn't guarantee their absence.
- Not "used by regulatory teams" — nobody would deploy a 4-drug weekend project; frame it as demonstrating a pattern (multi-source retrieval, conflict detection, citation verification) that generalizes to real compliance/cross-jurisdiction use cases, not as a deployed tool.
- Not "legally grounded" — it's grounded in retrieved source text, not legal analysis.
- Be ready to explain, in your own words, what the ReAct loop does, why the verifier is a separate step rather than folded into one prompt, and what the eval suite actually measures. If you can't explain a line of the pitch without notes, cut it.