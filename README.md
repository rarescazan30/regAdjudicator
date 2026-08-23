# RegAdjudicator

### FDA/EMA Regulatory Divergence Detection Agent

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active_development-orange.svg)

A reference implementation of an agentic RAG system that detects and explains conflicts between US (FDA) and EU (EMA) drug regulations — trial design requirements, approval pathways, patient population restrictions — and grounds every claim in a citation back to the source document.

> **Status:** In active development. See [Build Status](#build-status) below for what's working.

---

## What it does

Ask a question like *"How do FDA and EMA differ on surrogate endpoints for Alzheimer's therapies?"* and the agent:

1. Decides which regulatory sources it actually needs to check (not a fixed script — a US-only question won't trigger an EU lookup)
2. Retrieves relevant text from FDA and/or EMA source documents
3. Drafts a report with claims tied to specific source chunks
4. **Verifies its own draft** — a separate pass checks each cited claim against the actual retrieved text and flags or corrects anything unsupported before the answer is returned
5. Labels the outcome as `HARMONIZED`, `DIVERGENT`, or `INSUFFICIENT EVIDENCE`

## Why this exists

This is a proof-of-concept demonstrating an agentic system design pattern: dynamic tool selection, multi-source retrieval, self-correction (verify-and-retry), and evaluation-driven development. The FDA/EMA domain is the demo — the underlying pattern (compare two authoritative-but-conflicting sources, verify before answering) generalizes to any cross-jurisdiction compliance problem.

**What this is not:** a tool for actual regulatory decision-making. It covers 4 drugs and a handful of guidance documents — enough to demonstrate the architecture works, not enough (and not intended) to be relied on for real trial design or compliance decisions.

## Architecture

```
User question
     │
     ▼
Agent decides which tools to call (dynamic, not scripted)
     │
     ├─→ get_drug_metadata()      structured lookup (CSV)
     ├─→ search_fda()             semantic search, FDA labels + guidance
     └─→ search_ema()             semantic search, EMA SmPCs + guidance
     │
     ▼
Draft report, claims cited to specific source chunks
     │
     ▼
Verifier: checks every cited claim against the actual retrieved chunk text
     │
     ├─→ all claims supported → return report
     └─→ unsupported claim → retry with correction (max 2 attempts),
         then mark as "insufficient evidence" if still unresolved
```

## Agent guardrails

The agent operates under explicit constraints, enforced via system prompt:

- **Tools-not-memory:** never answer a factual/regulatory question from model memory — every claim must originate from a tool result.
- **Dual-jurisdiction check:** when evaluating a drug or trial design, check both FDA and EMA sources before concluding, unless the question is explicitly single-jurisdiction.
- **Explicit posture labeling:** every answer is labeled `HARMONIZED`, `DIVERGENT`, or `INSUFFICIENT EVIDENCE` — never left implicit.
- **Citation requirement:** every factual sentence must reference the specific source chunk it came from.
- **Retry cap:** the verifier can trigger at most 2 correction attempts before the agent reports a claim as unverifiable rather than continuing to guess.

<!-- ## Example trace *(this is a mock-up)*

```
Query: "Can we market Aducanumab for Alzheimer's in both the US and EU?"

[tool_call] get_drug_metadata(drug_name="aducanumab")
[result]    fda_status=Approved (Accelerated, 2021), ema_status=Refused (2021)

[tool_call] search_fda(query="aducanumab indication basis of approval")
[result]    chunk fda_adu_004: "...indicated based on reduction of amyloid
             beta plaques observed in treated patients..."

[tool_call] search_ema(query="aducanumab refusal assessment")
[result]    chunk ema_adu_002: "...the link between amyloid reduction and
             clinical benefit had not been established..."

[draft]     "...EMA rejected Aducanumab in 2022 citing safety concerns..."

[verify]    claim: "EMA rejected in 2022" → chunk ema_adu_002 shows 2021,
             and cites unproven efficacy, not safety → FAIL

[retry]     draft corrected: "EMA refused authorization in 2021, citing
             insufficient evidence that amyloid reduction translates to
             clinical benefit [ema_adu_002]"

[verify]    PASS — all claims supported

[output]    Posture: DIVERGENT. See full report below.
``` -->

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Google Gemini (`gemini-2.0-flash` via `google-genai` SDK, native tool calling) |
| Vector store | ChromaDB (local, persistent) |
| Validation | Pydantic |
| Evaluation | Custom trajectory + citation-grounding checks |

No LangChain/CrewAI — the agent loop is hand-written to keep the tool-calling mechanics explicit and explainable.

## Data sources

- **FDA:** [openFDA](https://open.fda.gov/) drug label API + FDA guidance documents (public, no API key required)
- **EMA:** Official Summary of Product Characteristics (SmPC) per product + EMA guidance documents (public)

4 drugs with real, documented FDA/EMA divergence:

| Drug | Indication | Divergence |
|---|---|---|
| Aducanumab (Aduhelm) | Alzheimer's | FDA: accelerated approval (surrogate endpoint) — EMA: refused |
| Lecanemab (Leqembi) | Early Alzheimer's | FDA: broad approval — EMA: restricted to ApoE ε4 non-carriers |
| Bevacizumab (Avastin) | Metastatic breast cancer | FDA: indication revoked — EMA: retained |
| Olaparib (Lynparza) | Prostate cancer | FDA: BRCA-mutated only — EMA: broader HRR-mutated population |

## Project structure

```
regadjudicator/
├── data/
│   ├── raw/                    # source PDFs/JSON (gitignored)
│   ├── processed/
│   │   └── drugs_metadata.csv
│   └── docs/
│       ├── drugs/               # {drug}_fda.md, {drug}_ema.md
│       └── guidance/            # cross-cutting FDA/EMA guidance docs
├── evals/
│   ├── evalset.json             # test cases: query, expected trajectory, required facts
│   └── run_evals.py
├── src/
│   ├── ingest.py
│   ├── retrieval.py
│   ├── tools.py
│   ├── agent.py
│   └── verify.py
├── demo.py
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- A Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))
- ~200MB disk space for the local ChromaDB store

## Running it

```bash
git clone https://github.com/<your-username>/regadjudicator.git
cd regadjudicator
pip install -r requirements.txt
cp .env.example .env   # add your Gemini API key
python demo.py
```

## Evaluation

`evals/run_evals.py` runs a set of test cases against the agent and checks:

- **Trajectory accuracy** — did it call the tools you'd expect for that question?
- **Classification accuracy** — did it correctly call the outcome harmonized/divergent/insufficient?
- **Grounding** — are required facts present and known hallucinations absent?

```bash
python evals/run_evals.py
```

<!-- Fill in once you've run it:
Current results: X/12 passing. See evals/evalset.json for individual cases.
-->

## Build status

- [x] Data ingestion (FDA + EMA, 4 drugs + guidance docs)
- [x] Retrieval layer (ChromaDB, chunking, search functions)
- [x] Tool schemas + agent ReAct loop
- [x] Verifier module
- [ ] Eval suite
- [ ] Demo CLI

## Limitations

- Covers 4 drugs and 2 guidance documents — not a comprehensive regulatory database
- Single-turn only, no session memory across questions
- Verification reduces unsupported claims; it does not guarantee their complete absence
- Not legal or regulatory advice

## Future directions

- Expand drug coverage and automate PDF ingestion
- Add multi-turn session state
- Generalize the architecture beyond pharma to other cross-jurisdiction compliance domains (e.g., tax, accessibility standards, engineering codes)