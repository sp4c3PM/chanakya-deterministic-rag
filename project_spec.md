# Chanakya OSINT Agent — Project Spec

**Owner:** Arijeet Roy  
**Version:** 0.1 (Day 3 of 90)  
**Working dir:** `/Users/arijeet.roy/Documents/chanakya/osint-agent/`

---

## What This Is

A retrieval-augmented intelligence tool for Indian defence procurement. It ingests policy documents, procurement records, and programme reports, then answers analyst questions with grounded, auditable responses.

This is not a chatbot. It is a **decision-support tool** — designed for users who have to put their signature on a decision and need to trace every answer back to a source document.

---

## The Core Design Principle: Deterministic Over Probabilistic

Consumer AI optimises for "good enough for most." A search result that satisfies 80% of users is a win.

Defence AI cannot work this way. When an MoD analyst asks "what is the offset obligation for this contract?", the answer cannot be approximate. If the system is uncertain, it must say so — not guess.

**This shapes every architectural decision below.**

---

## Architecture Decisions

### 1. Retrieval: TF-IDF (not dense embeddings)

**Decision:** Use TF-IDF for retrieval in v1.  
**Why:** Defence documents use precise, domain-specific terminology — "Buy (Indian-IDDM)", "AoN", "ToT". TF-IDF is exact-match on these terms. Dense embeddings may semantically conflate "Buy Indian" with "Buy Global" because they are contextually similar. In procurement, these are legally distinct categories.  
**Trade-off:** TF-IDF cannot handle paraphrasing or semantic similarity. Upgrade path: hybrid retrieval (TF-IDF + all-MiniLM) in v2, using Reciprocal Rank Fusion.  
**Current accuracy:** 96% retrieval, 92% answer on 25-question eval set.

### 2. Chunking: Structural with Section Breadcrumbs

**Decision:** Detect section headers (ALL CAPS lines, numbered clauses) and prepend them to every chunk.  
**Why:** Without headers, "The IC requirement is 50%" and "The IC requirement is 40%" look identical to TF-IDF — same keywords, different context. With the header prepended, the chunk reads "[BUY INDIAN-IDDM] The IC requirement is 50%" vs "[BUY INDIAN] The IC requirement is 40%". The model now has the label it needs to distinguish categories.  
**Impact:** Fixed Q18 (Buy Indian vs IDDM confusion) when added.

### 3. Merge Logic: Python Dict (not LLM)

**Decision:** Per-doc extraction → Python dict merge → conflict detection. No final LLM cleanup pass.  
**Why:** Three reasons:
1. **Determinism** — a Python dict merge is reproducible. An LLM merge is not.
2. **Auditability** — every field in the final output traces to a specific source doc. An LLM cleanup would obscure that lineage.
3. **Compute** — 8GB RAM, free-tier HF Space. An extra LLM call per fact sheet is too expensive.

**Conflict detection logic:** If a field has >1 unique value across source docs, it is flagged as a conflict. The system surfaces all values with their sources — it does not resolve the conflict. Resolution is the analyst's job.

### 4. Strict Null Policy

**Decision:** If a field is not found in a document, it is marked `NOT_SPECIFIED`. If not found across all documents, it is flagged `NOT IN CORPUS`.  
**Why:** In consumer products, "no results found" is a failure state. In defence intelligence, it is a valid and important data point. If the system cannot find the unit cost of a platform in its corpus, the honest answer is "this is not in our indexed documents" — not a hallucinated estimate.  
**PM logic:** A user who sees `NOT IN CORPUS` knows to check primary sources. A user who sees a hallucinated value may act on false data. The cost of the first is friction. The cost of the second is a procurement error.

### 5. Conflict as a Feature

**Decision:** Conflicts are surfaced to the user, not resolved silently.  
**Why:** Defence data is multidimensional. The BrahMos missile has at least two valid "costs": Rs 375 million (Philippines export deal) and Rs 18 crore (estimated domestic unit cost). Both are correct. They answer different questions. A system that picks one silently is less useful than one that shows both with context.  
**Implementation:** The `context` field in each extraction captures WHERE a value applies (e.g., "Philippines export contract, 2022" vs "domestic production estimate"). This turns a raw conflict into actionable intelligence.

---

## Atom Schema (v1)

The minimum viable fact sheet for any defence platform or programme:

| Field | Description | Example |
|-------|-------------|---------|
| `capability` | Performance parameters | Mach 2.8, range 290-450 km |
| `launch_platform` | Deployment mode | Air, Land, Sea |
| `indigenous_content` | IC% by value | 65% |
| `procurement_category` | DAP 2020 category | Buy (Indian-IDDM) |
| `unit_cost` | Cost per unit or contract value | Rs 375 million (export) |
| `context` | WHERE the value applies | Philippines export deal, 2022 |

**Why these 5+1 fields:** These are the parameters a procurement analyst needs to make a decision — capability (can it do the job?), platform (how is it deployed?), indigenisation (does it meet policy?), category (what rules apply?), cost (is it budgeted?), context (which deal/clause/date does this apply to?).

---

## Auditor's Trace Design

Every query response includes:
1. **VERDICT** — direct answer in one sentence
2. **CONFLICT ALERT** — if answer draws from >1 source document
3. **EVIDENCE CHAIN** — for each retrieved chunk:
   - Source document + section header (breadcrumb)
   - Relevance score (HIGH / MED / LOW)
   - Text snippet (first 200 chars)

**Design principle:** The user's job is to verify, not just to know. The trace makes verification possible without leaving the tool.

---

## Corpus (v1)

| Document | Type | Words | Status |
|----------|------|-------|--------|
| dap2020_sample.txt | Synthetic summary | 403 | Active |
| docs/dac_decisions.txt | Synthetic | 602 | Active |
| docs/defence_budget.txt | Synthetic | 501 | Active |
| docs/major_programmes.txt | Synthetic | 681 | Active |
| docs/drdo.txt | Synthetic | 273 | Active |
| docs/strategic_partner.txt | Synthetic | 313 | Active |
| docs/fdi_defence.txt | Synthetic | 293 | Active |
| docs/major_acquisitions.txt | Synthetic | 299 | Active |
| docs/dap2020_icmai.pdf | Real (ICMAI) | 6,635 | Active |
| docs/kpmg_defence_2047.pdf | Real (KPMG) | 12,758 | Active |
| docs/idsa_exports_2025.pdf | Real (IDSA) | 2,080 | Active |
| docs/large/hal_annual_2324.pdf | Real (BSE) | 145,717 | Parked (too large for TF-IDF) |

**Upgrade path:** HAL Annual Report needs selective section extraction before ingestion. Planned for Week 5 with PDF-to-Markdown conversion.

---

## Eval Framework

- **25-question hand-labeled eval set** (`eval_set.py`)
- Scored on: retrieval accuracy (correct source in top-k) + answer accuracy (keyword match)
- Run command: `CHANAKYA_MODEL=qwen2.5:1.5b .venv/bin/python eval.py`
- Current baseline: 96% retrieval / 92% answer

**Upgrade path (Week 5–6):**
- Scale to 50 questions
- Add LLM-as-judge scoring (faithfulness, groundedness) alongside keyword match
- Add conflict-detection test cases

---

## What's Not Built Yet (Planned)

| Feature | Phase | Signal for Sarvam |
|---------|-------|-------------------|
| Hybrid retrieval (TF-IDF + all-MiniLM) | Week 5 | Embeddings familiarity |
| Ontology tagging during ingestion | Week 5 | Ontology viewer |
| Multi-step agent with tool use | Week 7 | Agentic patterns |
| MCP server | Week 8 | MCP hands-on |
| Air-gapped mode (local Ollama) | Week 9 | On-prem deployment |
| Streamlit UI with audit panel | Week 10 | Product demo |
| Policy versioning / drift detection | Week 6 | Deterministic AI |

---

## Expert-in-the-Loop (EITL)

The final authority in a deterministic system is not the model or the document — it is the Subject Matter Expert.

When a Colonel looks at the BrahMos cost conflict and says "the Rs 18 crore figure is what we use for this mission", the system must remember that decision. This is the **Override Layer**.

**Implementation:**
- Conflicts are surfaced in the CLI with numbered choices
- Expert selects the correct value (or enters a custom one) + provides context
- Choice is saved to `overrides.json` with: value, context, override_by (name/role), override_date
- On all future queries for that entity+field, the override takes priority over any PDF-extracted value
- The override is clearly labelled in the output: `[OVERRIDE by Lt. Col Singh on 2026-04-26]`

**Why this matters:**
- Turns the system from a search engine into a growing intelligence
- The corpus improves with every expert interaction
- In air-gapped environments with no cloud sync, local expert knowledge is the only update mechanism

**Audit trail for corrections:** Every override is timestamped and attributed. The original conflicting values are preserved in the raw extraction — the override does not delete them, it takes priority.

---

## Data Provenance

**Problem:** How do we know a document in the corpus hasn't been tampered with?

In a consumer product, a modified document is a content bug. In a defence system, a tampered procurement document is a security incident.

**Implementation:**
- On every `load_docs()` call, SHA-256 checksums are computed for all ingested files
- Checksums are stored in `corpus_manifest.json`
- On subsequent loads, current checksums are compared against the manifest
- If any file has changed, a `[WARN] TAMPER DETECTED` alert is raised before processing begins

**Limitations of v1:**
- SHA-256 detects modification but not substitution (a fully replaced file with a valid hash would pass)
- No cryptographic signing of the manifest itself
- Upgrade path: GPG-signed manifest, immutable source logging (document ingestion date + URL + hash)

---

## Agentic Extensibility

Chanakya is not just a retrieval engine — it is a decision-support agent. The current system answers "What does the document say?" The agentic layer answers "What can the asset do?"

**The transition:**
- **Librarian mode (current):** "What is the range of BrahMos?" → retrieves from corpus → returns 290-450 km
- **Agent mode (next):** "Can a BrahMos launched from Jaisalmer reach Karachi?" → retrieves range from corpus → calls a geospatial distance tool → returns yes/no with margin

**Tool architecture:**
By exposing Python functions as tools via MCP (Model Context Protocol), the model can step outside the document corpus and into real-world calculations. The extracted Atom Schema fields (capability, launch_platform, unit_cost etc.) become the inputs to these tools.

**The three parameter extraction patterns (proven Day 3):**

| Pattern | Example query | Distance source | Speed/cost source | Calculation |
|---------|--------------|-----------------|-------------------|-------------|
| Corpus-dependent | "Can BrahMos reach Karachi from Jaisalmer?" | Haversine (Python) | Platform range from corpus (LLM) | Python |
| Question-native | "Can India afford 10 BrahMos at Rs 30 Cr each, budget Rs 250 Cr?" | — | All params from query (Python regex) | Python |
| Hybrid | "How long does BrahMos take to hit Karachi from Jaisalmer?" | Haversine (Python) | Mach from corpus (regex on chunks) | Python |

**Core principle:** The LLM is the Synthesizer, never the Calculator.
- LLM reads the document, identifies what's needed, and formats the final answer for an officer
- Python regex / Haversine / arithmetic handle every number
- This is not a limitation of 1.5B models — it is the correct design for any model size in a deterministic system

**4 tools live (Day 3):**
- `range_check(platform_range_km, origin, target)` — Haversine feasibility with margin
- `ic_compliance_check(ic_percent, procurement_category)` — DAP 2020 IC% table validation
- `budget_check(unit_cost_crore, quantity, budget_crore)` — procurement feasibility
- `calculate_impact_time(distance_km, mach_speed)` — time to impact in seconds and minutes

**MCP server design (Week 8):**
The MCP server exposes these tools to any compatible client (Claude Desktop, Cursor). Schema design is the PM's job — the tool descriptions must be precise enough that the model calls the right tool with the right parameters without hallucinating inputs.

**Proof of concept (Day 3, Week 1):** All 4 tools tested end-to-end with real queries. Range check (BrahMos to Karachi — IN RANGE, +1 km margin), budget check (10 BrahMos at Rs 30 Cr — OVER BUDGET by Rs 50 Cr), impact time (BrahMos Jaisalmer→Karachi — 467.5s / 7.79 min at Mach 2.8).

---

## Open Questions

1. Table extraction — DAP 2020 is 40% tables; pypdf returns word-soup for these. Need PDF-to-Markdown (e.g. pymupdf4llm) for Week 5.
2. Ontology seed schema — define entity types manually (platforms, actors, procurement stages, policy instruments) before LLM tagging at scale.
3. HAL Annual Report — 380 pages / 145k words. Need section-level extraction, not full ingest.
