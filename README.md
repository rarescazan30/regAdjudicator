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

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (tested on Python 3.14) |
| Agent LLM (Drafter) | Google Gemini (`gemini-3.5-flash-lite`, configurable to `gemini-3.7-flash`, via `google-genai` SDK for autonomous ReAct loop & multi-hop synthesis) |
| Verifier LLM (Auditor) | Google Gemini (`gemini-3.5-flash-lite` with Pydantic Constrained Decoding for sub-second deterministic audits) |
| Embeddings | `all-MiniLM-L6-v2` (Local ONNX dense vector embeddings, 384-dim) |
| Vector Store | ChromaDB (local, persistent, cosine distance with metadata jurisdiction isolation) |
| Structured Output | Pydantic v2 |
| Evaluation | Custom 12-case benchmark suite (`evals/run_evals.py`) measuring Trajectory, Posture Classification, and Grounding |
| Terminal UI | Rich |

*No LangChain or CrewAI — the agent control loop and verification state machines are hand-crafted in raw Python for full observability and zero token bloat.*

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
│   ├── config.py
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
git clone https://github.com/rarescazan30/regAdjudicator.git
cd regAdjudicator
pip install -r requirements.txt
cp .env.example .env   # add your Gemini API key
python demo.py
```

### Interactive Demonstration (Live Terminal Output)

```text
╭───────────────────────────────────────────────────────────────────────────────╮
│ regAdjudicator — FDA/EMA Regulatory Divergence Detection Agent                │
│ Agent Model: gemini-3.5-flash-lite | Verifier Model: gemini-3.5-flash-lite    │
│ Autonomous ReAct Loop • Two-Pass Grounding Verification • ChromaDB Vector RAG │
╰───────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────────────────────────────────╮
│  💡 Type your regulatory question  │  [1-5] Instant Presets  │  [A] Browse Scenarios  │  [Q] Quit  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────╯

Query / Option: 1

Selected preset [1]: Lecanemab (Leqembi) Divergence
Preset Query: Compare FDA and EMA approval scopes for Lecanemab (Leqembi). Did both agencies approve it and what restrictions exist regarding ApoE genotypes?

                          Agent Execution Summary                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                    ┃ Value                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Tool Trajectory           │ get_drug_metadata ➔ search_fda ➔ search_ema │
│ Citation Verification     │ ✓ PASSED (All Claims Grounded)              │
│ Self-Correction Retries   │ 0                                           │
│ Execution Latency         │ 12.30s                                      │
│ Retrieved Evidence Chunks │ 6                                           │
└───────────────────────────┴─────────────────────────────────────────────┘

╭────────────────────────────────────────────── Final Verified Regulatory Synthesis Report ──────────────────────────────────────────────╮
│                                                                                                                                        │
│   • Regulatory Posture: DIVERGENT                                                                                                      │
│   • Executive Summary: While both the US Food and Drug Administration (FDA) and the European Medicines Agency (EMA) ultimately         │
│     granted marketing authorization for lecanemab (Leqembi) for the treatment of early Alzheimer's disease, their approval scopes      │
│     diverge significantly regarding patient population restrictions based on genetic risk [lecanemab_ema_chunk_03]. The FDA approved   │
│     lecanemab for the broad early Alzheimer’s population regardless of ApoE genotype [lecanemab_fda_chunk_01], whereas the EMA         │
│     restricted authorization strictly to ApoE ε4 non-carriers and heterozygotes, explicitly excluding ApoE ε4 homozygotes              │
│     [lecanemab_ema_chunk_01].                                                                                                          │
│   • FDA Position & Evidence:                                                                                                           │
│      • Approved Indication: LEQEMBI is indicated for the treatment of Alzheimer's disease in patients with mild cognitive impairment   │
│        (MCI) or mild dementia stage of disease (early Alzheimer's disease), requiring confirmation of amyloid-beta pathology           │
│        [lecanemab_fda_chunk_01].                                                                                                       │
│      • Pathway: Traditional Approval.                                                                                                  │
│      • ApoE Genotype Scope: The FDA does not restrict prescribing based on ApoE ε4 status. Although testing is recommended prior to    │
│        initiation to inform risk discussions regarding Amyloid-Related Imaging Abnormalities (ARIA), homozygotes remain eligible for   │
│        treatment under US labeling [lecanemab_fda_chunk_01].                                                                           │
│      • Safety & Risk Management: Carries a Boxed Warning for ARIA (ARIA-E edema/effusion occurred in 12.6% and ARIA-H                  │
│        microhemorrhages/siderosis in 17.3% of treated patients) [lecanemab_fda_chunk_03]. While noting that ARIA incidence is          │
│        substantially higher in ApoE ε4 homozygotes (32.6% vs. 10.9% in heterozygotes and 5.4% in non-carriers), the FDA addressed      │
│        this via rigorous MRI monitoring protocols rather than a population contraindication [lecanemab_fda_chunk_01,                   │
│        lecanemab_fda_chunk_03].                                                                                                        │
│   • EMA Position & Evidence:                                                                                                           │
│      • Approved Indication (Restricted): Authorised for adult patients with early Alzheimer's disease who have 0 or 1 copy of the      │
│        ApoE ε4 allele (non-carriers or heterozygotes) [lecanemab_ema_chunk_01].                                                        │
│      • Pathway & CHMP Rationale: Following an initial negative CHMP opinion in July 2024—where regulators concluded that modest        │
│        clinical efficacy across the total population was outweighed by severe ARIA safety risks—the applicant requested a              │
│        re-examination with a sub-population analysis [lecanemab_ema_chunk_00]. This led to a revised positive opinion restricted to    │
│        patients with 0 or 1 ApoE ε4 allele [lecanemab_ema_chunk_00].                                                                   │
│      • ApoE Genotype Restrictions: Leqembi is strictly contraindicated / excluded in ApoE ε4 homozygotes (2 copies of the allele) in   │
│        the European Union [lecanemab_ema_chunk_01]. Pre-treatment genetic testing is mandatory [lecanemab_ema_chunk_01].               │
│      • Safety & Risk Management: Mandates a stringent Risk Management Plan, controlled distribution, clinician education, patient      │
│        alert cards, and serial pre-infusion MRI monitoring [lecanemab_ema_chunk_03].                                                   │
│   • Comparative Synthesis & Rationale:                                                                                                 │
│      • Divergence on Risk-Benefit Tolerance: The divergence between the FDA and EMA highlights differing regulatory philosophies       │
│        regarding benefit-risk assessments in neurodegenerative disease.                                                                │
│      • FDA Approach: The FDA prioritizes patient and physician autonomy within a shared-decision-making framework. It allows           │
│        high-risk patient subgroups (ApoE ε4 homozygotes) access to the therapy, provided they are fully informed of the markedly       │
│        elevated ARIA risks [lecanemab_fda_chunk_01] and monitored via mandatory MRI schedules [lecanemab_fda_chunk_03].                │
│      • EMA Approach: The EMA adopted a more conservative population-restriction strategy. Because ApoE ε4 homozygotes experience       │
│        disproportionately severe rates of ARIA-E and ARIA-H [lecanemab_fda_chunk_03], the CHMP concluded that the benefit-risk         │
│        balance is unfavorable for this specific genetic subgroup, leading to an explicit European exclusion [lecanemab_ema_chunk_01,   │
│        lecanemab_ema_chunk_03].                                                                                                        │
│                                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Evaluation

`evals/run_evals.py` executes a 12-case automated benchmark suite against the agent across 4 query categories:

- **Comparative Queries (4 cases):** Complex cross-jurisdictional synthesis (e.g., Aducanumab, Lecanemab, Bevacizumab, Olaparib).
- **Single-Jurisdiction Queries (6 cases):** Validates routing constraint handling (e.g., US FDA Accelerated Approval criteria, EMA CHMP refusal grounds).
- **Negative Controls & Edge Cases (2 cases):** Tests hallucination resistance against fictional compounds (`XYLOPHEN-99`) and unsupported off-label indications.

```bash
python -m evals.run_evals
```

### Benchmark Results

| Metric | Score | Target | Status |
|---|---|---|---|
| **Trajectory Adherence** | **100.0%** (12/12) | > 90% | Passed |
| **Citation Grounding Rate** | **100.0%** (12/12) | > 95% | Passed |
| **Posture Classification Precision** | **100.0%** (12/12) | > 90% | Passed |
| **Overall Case Pass Rate** | **100.0%** (12/12) | > 90% | Passed |
| **Average Query Latency** | **~12.5s** | < 20s | Passed |

## Build status

- [x] Data ingestion (FDA + EMA, 4 drugs + guidance docs)
- [x] Retrieval layer (ChromaDB, chunking, search functions)
- [x] Tool schemas + agent ReAct loop
- [x] Verifier module
- [x] Eval suite
- [x] Demo CLI

## Limitations

- Covers 4 drugs and 2 guidance documents — not a comprehensive regulatory database
- Single-turn only, no session memory across questions
- Verification reduces unsupported claims; it does not guarantee their complete absence
- Not legal or regulatory advice

## Future directions

- Expand drug coverage and automate PDF ingestion
- Add multi-turn session state
- Generalize the architecture beyond pharma to other cross-jurisdiction compliance domains (e.g., tax, accessibility standards, engineering codes)