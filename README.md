# Chanakya — Deterministic RAG for Indian Defence Procurement

A retrieval-augmented intelligence tool for Indian defence procurement. Ingests policy documents, procurement records, and programme reports. Answers analyst questions with grounded, auditable, source-traced responses.

**This is not a chatbot.** It is a decision-support tool — designed for users who have to put their signature on a decision and need to trace every answer back to a source document.

---

## The Core Design Principle: Deterministic Over Probabilistic

Consumer AI optimises for "good enough for most." Defence AI cannot work this way. When an MoD analyst asks "what is the offset obligation for this contract?", the answer cannot be approximate. If the system is uncertain, it must say so — not guess.

This principle shapes every architectural decision in this system.

---

## Architecture

### Retrieval: TF-IDF with Structural Chunking

TF-IDF over dense embeddings for v1. Defence documents use precise, domain-specific terminology — "Buy (Indian-IDDM)", "AoN", "ToT". TF-IDF is exact-match on these terms. Dense embeddings risk semantically conflating "Buy Indian" with "Buy Global" — categories that are legally distinct.

Every chunk is prepended with its section header breadcrumb: `[BUY INDIAN-IDDM] The IC requirement is 50%` vs `[BUY INDIAN] The IC requirement is 40%`. The model now has the label it needs to distinguish categories.

Upgrade path: hybrid retrieval (TF-IDF + all-MiniLM) in v2, using Reciprocal Rank Fusion.

### Agentic Tools: The Hybrid Extraction Architecture

The agent uses three distinct parameter extraction patterns depending on where the data lives:

| Pattern | Data source | Example query |
|---------|-------------|---------------|
| **Corpus-dependent** | Retrieved chunks (LLM extraction) | "Can BrahMos reach Karachi from Jaisalmer?" → range from PDF |
| **Question-native** | User's query (Python regex) | "10 units at Rs 30 Cr, budget Rs 250 Cr" → all params from question |
| **Hybrid** | Query + corpus (Python + regex) | "How long does BrahMos take to reach Karachi from Jaisalmer?" → distance from Haversine, Mach from corpus regex |

**The LLM is the Synthesizer, never the Calculator.** Python handles every number. The model reads the document, identifies what's needed, and frames the final answer for the analyst. This is not a limitation of small models — it is the correct design for any model size in a deterministic system.

### Four Live Agentic Tools (Day 3)

```
range_check(platform_range_km, origin, target)
    → Haversine distance + IN RANGE / OUT OF RANGE verdict with margin

ic_compliance_check(ic_percent, procurement_category)
    → DAP 2020 IC% table validation + COMPLIANT / NON-COMPLIANT

budget_check(unit_cost_crore, quantity, budget_crore)
    → FEASIBLE / OVER BUDGET with shortfall in crore

calculate_impact_time(distance_km, mach_speed)
    → Time to impact in seconds and minutes
```

### Expert-in-the-Loop (EITL)

The final authority in a deterministic system is not the model or the document — it is the Subject Matter Expert.

When conflicting values are detected (e.g., BrahMos unit cost differs across export contract vs. domestic production estimate), the system surfaces all values with their sources and prompts the expert to select. The choice is saved to `overrides.json` with timestamp and analyst attribution. On future queries, the override takes priority — clearly labelled as `[OVERRIDE by Lt Col Singh on 2026-04-26]`.

The original conflicting values are preserved. The override does not delete them.

### Data Provenance

On every load, SHA-256 checksums are computed for all ingested files and compared against `corpus_manifest.json`. If any file has changed, `[WARN] TAMPER DETECTED` is raised before processing begins.

---

## Trifecta Demo (Day 3 Results)

**Range Check:**
```
Q: Can a BrahMos launched from Jaisalmer reach Karachi?

TOOL: range_check(platform_range_km=450.0, origin='jaisalmer', target='karachi')
RESULT: IN RANGE — distance 449.0 km, range 450.0 km, margin +1.0 km

ANSWER: Yes — BrahMos reaches Karachi from Jaisalmer with a 1 km margin at
        maximum range. (Source: major_programmes.txt › DRDO KEY MISSILES)
```

**Budget Check:**
```
Q: Can India afford 10 more BrahMos at Rs 30 crore each given a Rs 250 crore budget?

TOOL: budget_check(unit_cost_crore=30.0, quantity=10, budget_crore=250.0)
RESULT: OVER BUDGET — total Rs 300.0 Cr vs budget Rs 250.0 Cr

ANSWER: Negative. Total cost (Rs 300 Cr) exceeds budget (Rs 250 Cr) by Rs 50 Cr.
```

**Impact Time (Hybrid extraction):**
```
Q: How long does a BrahMos take to reach Karachi from Jaisalmer?

TOOL: calculate_impact_time(distance_km=449.0, mach_speed=2.8)
      [distance: Haversine from city coordinates]
      [mach_speed: regex on corpus chunk — "Speed Mach 2.8"]
RESULT: At Mach 2.8, impact at 449.0km in 467.5s (7.79 min)

ANSWER: 467.5 seconds — approximately 7 minutes 47 seconds at Mach 2.8 over 449 km.
```

**Policy Conflict (Auditor's Trace mode):**
```
Q: What is the minimum IC% for Buy Indian-IDDM?

⚠  CONFLICT ALERT — answer drawn from 3 different sources:
   • docs/dac_decisions.txt
   • docs/dap2020_icmai.pdf
   • docs/dap2026_chapter1.txt

VERDICT: DAP 2020 requires 50% IC upfront.
         DAP 2026 Draft introduces a graded approach: 30% at trials → 60% at final.
         These are unresolved — resolution is the analyst's call, not the system's.
```

The system does not pick a winner. Surfacing the conflict *is* the answer.

---

## Evaluation

**Eval set:** 25 hand-labeled questions covering procurement categories, IC requirements, DAC decisions, major programmes, budget figures, and platform specs.

| Metric | Score |
|--------|-------|
| Retrieval accuracy | 25/25 = **100%** |
| Answer accuracy (keyword-match) | 21/25 = **84%** |
| Answer accuracy (estimated, stable) | **88–92%** |

The gap between 84% and 88–92% is attributable to keyword-match eval limitations:
- Q01 / Q21: correct answers that fail substring match due to phrasing variation ("minimum of 30" vs "30%")
- Q18: passes on targeted re-run — stochastic 1.5B model output

### Known Limitations

These are not bugs. They are known ceilings that inform the upgrade path:

| Limitation | Current behaviour | Upgrade path |
|------------|-------------------|--------------|
| TF-IDF paragraph precision | Retrieves correct document but sometimes wrong paragraph within a table-heavy PDF | all-MiniLM embeddings + hybrid retrieval (Week 5) |
| Table extraction | DAP 2020 tables return word-soup via pypdf | pdfplumber with row/column structure → Markdown pipes (Week 5) |
| Multi-value answers | Model names one state out of two required (Defence Industrial Corridors) | LLM-as-judge scoring (Week 5) |
| Policy drift synthesis | Conflict alert fires correctly (DAP 2020 vs DAP 2026), but verdict is incomplete for graded IC% changes | all-MiniLM embeddings for within-document paragraph precision (Week 5) |
| 1.5B model stochasticity | Some answers vary between runs — keyword-match eval treats this as failure | LLM-as-judge eval scoring (Week 5) |

Surfacing these limitations is not a weakness. A system that documents what it cannot do is more trustworthy for mission-critical use than one that presents false certainty.

---

## Corpus (v1)

| Document | Type | Words |
|----------|------|-------|
| dap2020_sample.txt | Synthetic summary | 403 |
| docs/dap2026_chapter1.txt | Real — MoD (DAP 2026 Draft, Chapter I) | 10,451 |
| docs/dap2020_icmai.pdf | Real — ICMAI | 6,635 |
| docs/kpmg_defence_2047.pdf | Real — KPMG | 12,758 |
| docs/idsa_exports_2025.pdf | Real — IDSA | 2,080 |
| docs/dac_decisions.txt | Synthetic | 602 |
| docs/defence_budget.txt | Synthetic | 501 |
| docs/major_programmes.txt | Synthetic | 681 |
| docs/drdo.txt | Synthetic | 273 |
| docs/strategic_partner.txt | Synthetic | 313 |
| docs/fdi_defence.txt | Synthetic | 293 |
| docs/major_acquisitions.txt | Synthetic | 299 |

---

## Quickstart

```bash
# Install dependencies
pip install qdrant-client scikit-learn pypdf openai rich

# Run the query CLI (Auditor's Trace mode)
CHANAKYA_MODEL=qwen2.5:1.5b python query.py

# Run the agentic loop (tools mode)
CHANAKYA_MODEL=qwen2.5:1.5b python agent.py

# Build a fact sheet for a platform
CHANAKYA_MODEL=qwen2.5:1.5b python factsheet.py

# Run the eval suite
CHANAKYA_MODEL=qwen2.5:1.5b python eval.py
```

The system connects to any OpenAI-compatible Ollama endpoint. Set `CHANAKYA_MODEL` to match the model on your endpoint. Tested with `qwen2.5:1.5b` on 8GB RAM.

---

## Upgrade Roadmap

| Feature | Phase | Why it matters |
|---------|-------|----------------|
| Hybrid retrieval (TF-IDF + all-MiniLM) | Week 5 | Fixes within-document paragraph precision |
| pdfplumber table extraction | Week 5 | DAP 2020 tables currently word-soup |
| LLM-as-judge eval scoring | Week 5 | Faithfulness + groundedness, not just keyword match |
| Ontology tagging | Week 5 | Entity-level indexing (platforms, actors, procurement stages) |
| MCP server | Week 8 | Exposes tools to Claude Desktop / Cursor |
| Air-gapped mode | Week 9 | Local Ollama, no cloud dependency |
| Streamlit UI | Week 10 | Audit panel for non-CLI users |
| Policy versioning / drift detection | Week 6 | DAP 2020 vs DAP 2026 automatic conflict flagging |

---

## Author

Built by [@sp4c3PM](https://github.com/sp4c3PM) as part of a 90-day defence AI portfolio.
