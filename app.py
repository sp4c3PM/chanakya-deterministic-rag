"""
Chanakya C2 — Command & Control Dashboard
Streamlit interface showing the full deterministic trace:
  Retrieved Chunk → LLM Extraction → Python Normalization → Final Logic
"""
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass

import os
import re
import json
import openai
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Chanakya C2",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tactical dark theme ──────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0a0f1e; color: #e0e0e0; font-family: 'Courier New', monospace; }
  .stTextInput > div > div > input {
    background-color: #0d1a2d; color: #e0e0e0;
    border: 1px solid #f0a500; border-radius: 4px; font-family: monospace;
  }
  .stButton > button {
    background-color: #f0a500 !important; color: #0a0f1e !important;
    font-weight: bold; border: none; border-radius: 4px;
    font-family: monospace; letter-spacing: 1px;
  }
  .stButton > button:hover { background-color: #ffb930 !important; }
  .verdict-ok  { color: #00c853; font-weight: bold; font-size: 1.3em; }
  .verdict-warn { color: #ff4444; font-weight: bold; font-size: 1.3em; }
  .trace-step {
    background: #0d1a2d; border-left: 3px solid #f0a500;
    padding: 8px 14px; margin: 6px 0; font-family: monospace; font-size: 0.82em;
    border-radius: 0 4px 4px 0;
  }
  .conflict-box {
    background: #1a0505; border: 1px solid #ff4444;
    border-radius: 4px; padding: 12px 16px; margin: 10px 0;
  }
  .qualifier-box {
    background: #1a1200; border: 1px solid #f0a500;
    border-radius: 4px; padding: 10px 14px; margin: 10px 0;
  }
  .stExpander { border: 1px solid #1e3050 !important; border-radius: 4px; }
  h1, h2, h3 { color: #f0a500; font-family: monospace; }
  hr { border-color: #1e3050; }
  .stMetric { background: #0d1a2d; border-radius: 6px; padding: 8px; }
  .stAlert { background: #0d1a2d; }
</style>
""", unsafe_allow_html=True)

# ── Constants (imported from agent/ingest to stay DRY) ───────────────────────
HF_OLLAMA_ENDPOINT = "https://gtf330-ollama-test.hf.space/v1"
MODEL = os.environ.get("CHANAKYA_MODEL", "tinyllama")
COLLECTION = "defence_docs"

# ── Index (built once, cached across reruns) ─────────────────────────────────
@st.cache_resource(show_spinner="Building corpus index...")
def load_index():
    import sys, io
    # Capture ingest stdout so we can show it in sidebar
    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    from ingest import load_docs, build_index
    chunks = load_docs()
    qdrant, vectorizer = build_index(chunks)
    sys.stdout = old_out
    n_docs = len(set(c["source"] for c in chunks))
    return qdrant, vectorizer, chunks, n_docs, buf.getvalue()

# ── Agent execution with trace capture ───────────────────────────────────────
def run_traced(question: str, qdrant, vectorizer) -> dict:
    """
    Run one agent turn and return a structured trace dict.
    Mirrors react_loop() in agent.py but captures each step for UI rendering.
    """
    from agent import (
        route_query, pre_extract_params, extract_tool_call,
        get_tool_prompt, TOOL_MAP,
    )

    trace = {
        "query":            question,
        "routing":          None,
        "tool_name":        None,
        "chunks":           [],
        "extraction_method": None,
        "raw_llm_output":   None,
        "qualifier":        None,
        "params":           None,
        "tool_result":      None,
        "answer":           None,
        "conflict_detected": False,
        "conflict_sources": [],
    }

    # Step 1 — route
    tool_name = route_query(question)
    trace["routing"] = tool_name
    trace["tool_name"] = tool_name

    # Step 2 — retrieve
    vec = vectorizer.transform([question]).toarray()[0]
    points = qdrant.query_points(collection_name=COLLECTION, query=vec.tolist(), limit=4).points
    trace["chunks"] = [
        {
            "header": p.payload.get("header", ""),
            "source": Path(p.payload["source"]).name,
            "text":   p.payload["text"][:800],
        }
        for p in points
    ]

    # Conflict detection: 3+ distinct sources for the same query
    sources = list(dict.fromkeys(Path(p.payload["source"]).name for p in points))
    if len(sources) >= 3:
        trace["conflict_detected"] = True
        trace["conflict_sources"]  = sources

    context = "\n\n".join([
        f"[{p.payload.get('header', p.payload['source'])}]\n{p.payload['text'][:800]}"
        for p in points
    ])

    llm = openai.OpenAI(api_key="ollama", base_url=HF_OLLAMA_ENDPOINT, timeout=60)

    # Step 3a — try Python-side pre-extraction
    pre_params = pre_extract_params(tool_name, question) if tool_name != "retrieval_only" else None

    # Hybrid path: distance from Haversine, mach from corpus regex
    if pre_params and any(v is None for k, v in pre_params.items() if not k.startswith("_")):
        if tool_name == "calculate_impact_time":
            origin = pre_params.pop("_origin", "?")
            target = pre_params.pop("_target", "?")
            full_text = " ".join(p.payload["text"] for p in points)
            mach_m = re.search(r'[Mm]ach\s+(\d+(?:\.\d+)?)', full_text)
            if mach_m:
                pre_params["mach_speed"] = float(mach_m.group(1))
                trace["extraction_method"] = f"hybrid — distance: Haversine ({origin}→{target}), mach: corpus regex"
            else:
                pre_params = None

    if pre_params and all(v is not None for k, v in pre_params.items() if not k.startswith("_")):
        clean = {k: v for k, v in pre_params.items() if not k.startswith("_")}
        fn = TOOL_MAP.get(tool_name)
        if fn:
            try:
                tool_result = fn(**clean)
                trace["extraction_method"] = trace["extraction_method"] or "pre-extract (Python regex on question)"
                trace["params"]      = clean
                trace["tool_result"] = tool_result
                r2 = llm.chat.completions.create(
                    model=MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": "You are a defence analyst. Summarize the tool result in one clear sentence for a senior officer."},
                        {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}\n\nOriginal question: {question}"},
                    ]
                )
                trace["answer"] = r2.choices[0].message.content.strip()
                return trace
            except Exception as e:
                trace["answer"] = f"Tool error: {e}"
                return trace

    # Step 3b — LLM extraction path
    qualifier_rule = (
        '\n- If a number has a qualifier ("up to", "not exceeding", "approximately", "at least"), '
        'add a "qualifier" field to your JSON with that qualifier string. Never strip a caveat silently.'
    )
    tool_prompt = (get_tool_prompt(tool_name) + qualifier_rule) if tool_name != "retrieval_only" else ""

    system = (
        "You are a defence procurement analyst and tactical advisor. "
        "Answer using ONLY the provided context. "
        "Extract exact numbers before using any tool. "
        "If the answer is not in the context, say NOT_FOUND.\n"
        + tool_prompt
    )

    r1 = llm.chat.completions.create(
        model=MODEL, max_tokens=200,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
    )
    r1_text = r1.choices[0].message.content.strip()
    trace["raw_llm_output"] = r1_text

    tool_call = extract_tool_call(r1_text) if tool_name != "retrieval_only" else None

    if tool_call:
        params = dict(tool_call.get("params", {}))
        trace["qualifier"] = tool_call.get("qualifier") or params.pop("qualifier", None)

        if any(v is None for v in params.values()):
            trace["answer"] = "NOT_FOUND — model could not extract required parameters from context."
            trace["extraction_method"] = "llm (failed — null params)"
            return trace

        fn = TOOL_MAP.get(tool_call["tool"])
        if fn:
            try:
                tool_result = fn(**params)
                trace["extraction_method"] = "llm"
                trace["params"]      = params
                trace["tool_result"] = tool_result
                r2 = llm.chat.completions.create(
                    model=MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": "You are a defence analyst. Summarize the tool result in one clear sentence for a senior officer."},
                        {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}\n\nOriginal question: {question}"},
                    ]
                )
                trace["answer"] = r2.choices[0].message.content.strip()
                return trace
            except Exception as e:
                trace["answer"] = f"Tool error: {e}"
                return trace

    # Retrieval-only answer
    trace["extraction_method"] = "retrieval-only (no tool)"
    trace["answer"] = r1_text
    return trace

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ CHANAKYA C2")
st.markdown("**Deterministic Defence Intelligence · Arithmetic Wall Active**")
st.markdown("---")

# Load index
try:
    qdrant, vectorizer, corpus_chunks, n_docs, load_log = load_index()
except Exception as e:
    st.error(f"Index load failed: {e}")
    st.stop()

# Sidebar — corpus status
with st.sidebar:
    st.markdown("### Corpus Status")
    st.metric("Documents", n_docs)
    st.metric("Chunks", len(corpus_chunks))
    st.metric("Model", MODEL)
    st.markdown("**Endpoint:** HF Space (qwen2.5:1.5b)")
    st.markdown("---")
    st.markdown("### Arithmetic Wall")
    st.markdown("""
    The LLM is **forbidden** from doing math.

    - Range → Python Haversine
    - Budget → Python arithmetic
    - Impact time → ISA physics (299 m/s @ 10km)
    - Units → Static rate table (INR Cr)
    - IC% → DAP threshold table or RAG-retrieved
    """)
    with st.expander("Ingestion log"):
        st.code(load_log or "(empty)", language=None)

# Query bar
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "QUERY",
        placeholder="Can BrahMos reach Karachi from Jaisalmer?",
        label_visibility="collapsed",
    )
with col2:
    submit = st.button("EXECUTE", use_container_width=True)

with st.expander("Example queries"):
    st.markdown("""
    - `Can a BrahMos launched from Jaisalmer reach Karachi?`
    - `How long does BrahMos take to reach Karachi from Jaisalmer?`
    - `Can India afford 10 BrahMos at Rs 30 crore each with a Rs 250 crore budget?`
    - `What is the minimum IC% for Buy Indian-IDDM?`
    - `What is the offset obligation for Buy Global contracts?`
    """)

if not (submit and question):
    st.stop()

# ── Execute ───────────────────────────────────────────────────────────────────
with st.spinner("Routing → Retrieving → Extracting → Executing..."):
    trace = run_traced(question, qdrant, vectorizer)

st.markdown("---")

# Status row
c1, c2, c3 = st.columns(3)
c1.metric("Router",     trace["routing"].replace("_", " ").title())
c2.metric("Extraction", trace["extraction_method"] or "—")
c3.metric("Sources",    len(trace["chunks"]))

# Qualifier warning
if trace.get("qualifier"):
    st.markdown(f"""
    <div class="qualifier-box">
    ⚠ <b>QUALIFIER DETECTED:</b> &ldquo;{trace['qualifier']}&rdquo;<br>
    The extracted value may be a limit or estimate, not a fixed fact.
    Verify against the source document before signing off.
    </div>
    """, unsafe_allow_html=True)

# Conflict alert
if trace["conflict_detected"]:
    src_list = "  ·  ".join(trace["conflict_sources"])
    st.markdown(f"""
    <div class="conflict-box">
    ⚠ <b>CONFLICT ALERT</b> — answer drawn from {len(trace['conflict_sources'])} sources.<br>
    The system surfaces all values; resolution is the analyst's call.<br>
    <code>{src_list}</code>
    </div>
    """, unsafe_allow_html=True)

# Verdict + answer
st.markdown("### VERDICT")
if trace["tool_result"]:
    verdict = trace["tool_result"].get("verdict", "")
    positive = any(w in verdict for w in ["IN RANGE", "FEASIBLE", "COMPLIANT"])
    css = "verdict-ok" if positive else "verdict-warn"
    st.markdown(f'<div class="{css}">{verdict}</div>', unsafe_allow_html=True)
    st.markdown("")

st.info(trace["answer"] or "No answer generated.")

# Trifecta metrics
if trace["tool_result"]:
    tr = trace["tool_result"]
    st.markdown("### TACTICAL METRICS")
    m1, m2, m3 = st.columns(3)

    with m1:
        if "distance_km" in tr:
            st.metric("Distance", f"{tr['distance_km']} km")
        elif "required_ic_percent" in tr:
            st.metric("Required IC%", f"{tr['required_ic_percent']}%")
        elif "total_cost_crore" in tr:
            st.metric("Total Cost", f"₹{tr['total_cost_crore']} Cr")

    with m2:
        if "time_seconds" in tr:
            st.metric("Impact Time", f"{tr['time_seconds']} s  ({tr['time_minutes']} min)")
        elif "actual_ic_percent" in tr:
            st.metric("Actual IC%", f"{tr['actual_ic_percent']}%")
        elif "budget_crore" in tr:
            st.metric("Budget", f"₹{tr['budget_crore']} Cr")

    with m3:
        if "margin_km" in tr:
            st.metric("Range Margin", f"{tr['margin_km']:+} km")
        elif "compliant" in tr:
            st.metric("Status", "COMPLIANT ✓" if tr["compliant"] else "NON-COMPLIANT ✗")
        elif "shortfall_crore" in tr and tr["shortfall_crore"]:
            st.metric("Shortfall", f"₹{tr['shortfall_crore']} Cr")

    # Physics note for impact time
    if "speed_of_sound_ms" in tr:
        st.caption(
            f"Speed of sound: {tr['speed_of_sound_ms']} m/s at {tr.get('altitude_m', 10000)/1000:.0f} km altitude "
            f"(ISA standard atmosphere). Sea-level 343 m/s would understate impact time by ~14.5%."
        )

# ── Audit Trail ───────────────────────────────────────────────────────────────
with st.expander("🔍 AUDIT TRAIL — Deterministic Trace"):
    st.markdown("#### Step 1 — Retrieved Chunks")
    for i, chunk in enumerate(trace["chunks"], 1):
        label = f"[{chunk['source']}]" + (f" › {chunk['header']}" if chunk["header"] else "")
        st.markdown(f"""
        <div class="trace-step">
        <b>[{i}] {label}</b><br>{chunk['text'][:400]}…
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Step 2 — LLM Extraction")
    st.markdown(f"`Method: {trace['extraction_method']}`")
    if trace["raw_llm_output"]:
        st.code(trace["raw_llm_output"], language="json")
    else:
        st.markdown("*Python pre-extraction — no LLM call for param extraction*")

    st.markdown("#### Step 3 — Python Normalization")
    if trace["params"]:
        st.code(json.dumps(trace["params"], indent=2), language="json")
        st.caption("All arithmetic from this point is deterministic Python. The LLM does not touch these numbers.")
    else:
        st.markdown("*No tool params — retrieval-only answer*")

    st.markdown("#### Step 4 — Tool Result")
    if trace["tool_result"]:
        st.code(json.dumps(trace["tool_result"], indent=2), language="json")
    else:
        st.markdown("*No tool executed*")
