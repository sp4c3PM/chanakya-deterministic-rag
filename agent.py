"""
Agentic query loop — ReAct pattern, manual JSON, router approach.
No LangChain. The router classifies the query first, loads only the
relevant tool description, then lets the LLM decide whether to call it.
"""
try:
    import truststore; truststore.inject_into_ssl()  # corporate SSL proxy (optional)
except ImportError:
    pass
import os
import json
import re
import openai
from rich.console import Console
from rich.panel import Panel
from rich import box
from ingest import load_docs, build_index, COLLECTION
from tools import TOOLS, range_check, ic_compliance_check, budget_check, calculate_impact_time

HF_OLLAMA_ENDPOINT = "https://gtf330-ollama-test.hf.space/v1"
MODEL = os.environ.get("CHANAKYA_MODEL", "tinyllama")
console = Console()

TOOL_MAP = {t["name"]: t["fn"] for t in TOOLS}

# Router: keyword-based, no LLM call needed
# Order matters: more specific patterns checked first
def route_query(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["impact", "seconds", "how long", "how many minutes", "arrive", "time to"]):
        return "calculate_impact_time"
    if any(w in q for w in ["ic%", "indigenous content", "compliant", "iddm", "comply"]):
        return "ic_compliance_check"
    if any(w in q for w in ["budget", "afford", "crore", "feasible"]):
        return "budget_check"
    if any(w in q for w in ["range", "reach", "hit", "distance", "km away"]):
        return "range_check"
    return "retrieval_only"

TOOL_EXAMPLES = {
    "range_check": '{"tool": "range_check", "params": {"platform_range_km": 450.0, "origin": "jaisalmer", "target": "karachi"}}',
    "calculate_impact_time": '{"tool": "calculate_impact_time", "params": {"distance_km": 520.0, "mach_speed": 2.8}}',
    "ic_compliance_check": '{"tool": "ic_compliance_check", "params": {"ic_percent": 55.0, "procurement_category": "Buy Indian-IDDM"}}',
    "budget_check": '{"tool": "budget_check", "params": {"unit_cost_crore": 30.0, "quantity": 10, "budget_crore": 350.0}}',
}

def get_tool_prompt(tool_name: str) -> str:
    tool = next((t for t in TOOLS if t["name"] == tool_name), None)
    if not tool:
        return ""
    params = "\n".join([f"  - {k}: {v}" for k, v in tool["parameters"].items()])
    example = TOOL_EXAMPLES.get(tool_name, "")
    return f"""
You have access to one tool: {tool["name"]}
Description: {tool["description"]}
Parameters:
{params}

Rules for extracting parameters:
- Extract numeric values as floats (e.g. "290 km extended to 450 km" → use 450.0, the maximum)
- Location names must be lowercase single words (e.g. "Jaisalmer" → "jaisalmer")
- NEVER use null — if a value is not in the context, output NOT_FOUND instead of calling the tool

Respond with ONLY this JSON (no other text, no explanation):
{example}

If the required numbers are NOT in the context, respond: NOT_FOUND
"""

def extract_tool_call(text: str):
    # Find the outermost JSON object containing "tool" — handles nested braces
    for m in re.finditer(r'\{', text):
        start = m.start()
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if "tool" in parsed and "params" in parsed:
                            return parsed
                    except Exception:
                        pass
                    break
    return None

def get_client():
    return openai.OpenAI(api_key="ollama", base_url=HF_OLLAMA_ENDPOINT, timeout=60)

def retrieve(qdrant, vectorizer, query, top_k=4):
    vec = vectorizer.transform([query]).toarray()[0]
    return qdrant.query_points(collection_name=COLLECTION, query=vec.tolist(), limit=top_k).points

def _nums(text):
    """Extract all floats/ints from text."""
    return [float(x.replace(",", "")) for x in re.findall(r'\d[\d,]*(?:\.\d+)?', text)]

def pre_extract_params(tool_name: str, question: str) -> dict | None:
    """
    Python-side extraction for queries where parameters are explicit in the question.
    Returns a params dict if all required params found, else None (falls back to LLM).
    """
    q = question.lower()
    nums = _nums(question)

    if tool_name == "budget_check":
        # Patterns: "X units/more/missiles at Rs Y crore each, budget Rs Z crore"
        quantity = None
        unit_cost = None
        budget = None

        qty_m = re.search(r'(\d+)\s+(?:more|units?|missiles?|aircraft|helicopters?)', q)
        if qty_m:
            quantity = int(qty_m.group(1))

        # "Rs Y crore each" or "at Y crore"
        uc_m = re.search(r'(?:rs\s*)?(\d[\d,]*(?:\.\d+)?)\s*crore\s*(?:each|per unit)', q)
        if uc_m:
            unit_cost = float(uc_m.group(1).replace(",", ""))

        # "budget of Rs Z crore" or "Rs Z crore budget"
        bud_m = re.search(r'(?:rs\s*)?(\d[\d,]*(?:\.\d+)?)\s*crore\s*budget', q)
        if not bud_m:
            bud_m = re.search(r'budget\s+(?:of\s+)?(?:rs\s*)?(\d[\d,]*(?:\.\d+)?)\s*crore', q)
        if bud_m:
            budget = float(bud_m.group(1).replace(",", ""))

        if quantity and unit_cost and budget:
            return {"unit_cost_crore": unit_cost, "quantity": quantity, "budget_crore": budget}

    if tool_name == "calculate_impact_time":
        # Case 1: explicit km + mach in question
        dist_m = re.search(r'(\d[\d,]*(?:\.\d+)?)\s*km', q)
        mach_m = re.search(r'mach\s+(\d+(?:\.\d+)?)', q)
        if dist_m and mach_m:
            return {"distance_km": float(dist_m.group(1)), "mach_speed": float(mach_m.group(1))}
        # Case 2: city names in question → compute distance from LOCATIONS; mach needs corpus
        from tools import LOCATIONS, haversine_km
        found = [(name, coords) for name, coords in LOCATIONS.items() if name in q]
        if len(found) >= 2:
            (n1, c1), (n2, c2) = found[0], found[1]
            dist = round(haversine_km(*c1, *c2), 1)
            # Return partial params — mach_speed is None, handled by hybrid extraction
            return {"distance_km": dist, "mach_speed": None, "_origin": n1, "_target": n2}

    return None

def react_loop(llm, qdrant, vectorizer, question: str):
    tool_name = route_query(question)
    console.print(f"  [dim]Router → {tool_name}[/dim]")

    # Step 1: retrieve context
    chunks = retrieve(qdrant, vectorizer, question)
    context = "\n\n".join([
        f"[{c.payload.get('header', c.payload['source'])}]\n{c.payload['text'][:300]}"
        for c in chunks
    ])

    # Step 1b: try Python-side param extraction (budget, impact time — numbers in question)
    pre_params = pre_extract_params(tool_name, question) if tool_name != "retrieval_only" else None

    # Hybrid extraction: some params from Python (distance), some from corpus (regex)
    if pre_params and any(v is None for k, v in pre_params.items() if not k.startswith("_")):
        missing = [k for k, v in pre_params.items() if v is None and not k.startswith("_")]
        if missing and tool_name == "calculate_impact_time":
            origin = pre_params.pop("_origin", "?")
            target = pre_params.pop("_target", "?")
            dist = pre_params["distance_km"]
            console.print(f"  [dim]Hybrid: distance={dist}km ({origin}→{target}), regex-extracting {missing} from corpus[/dim]")
            # Regex extraction on retrieved chunks — deterministic, no LLM hallucination risk
            full_context_text = " ".join(c.payload["text"] for c in chunks)
            mach_m = re.search(r'[Mm]ach\s+(\d+(?:\.\d+)?)', full_context_text)
            if mach_m:
                pre_params["mach_speed"] = float(mach_m.group(1))
                console.print(f"  [dim]Corpus regex: mach_speed={pre_params['mach_speed']}[/dim]")
            else:
                pre_params = None  # Can't complete — fall through to full LLM path

    if pre_params and all(v is not None for k, v in pre_params.items() if not k.startswith("_")):
        # Strip internal keys before calling tool
        clean_params = {k: v for k, v in pre_params.items() if not k.startswith("_")}
        fn = TOOL_MAP.get(tool_name)
        if fn:
            try:
                tool_result = fn(**clean_params)
                tool_call = {"tool": tool_name, "params": clean_params}
                console.print(f"  [dim]Pre-extracted: {tool_name}({clean_params})[/dim]")
                r2 = llm.chat.completions.create(
                    model=MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": "You are a defence analyst. Summarize the tool result in one clear sentence for an officer."},
                        {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}\n\nOriginal question: {question}"}
                    ]
                )
                return r2.choices[0].message.content.strip(), chunks, tool_call, tool_result
            except Exception as e:
                console.print(f"  [dim]Pre-extract tool error: {e}[/dim]")

    tool_prompt = get_tool_prompt(tool_name) if tool_name != "retrieval_only" else ""

    system = (
        "You are a defence procurement analyst and tactical advisor. "
        "Answer using ONLY the provided context. "
        "Extract exact numbers before using any tool. "
        "If the answer is not in the context, say NOT_FOUND.\n"
        + tool_prompt
    )

    # Step 2: first LLM call — may return a tool call or a direct answer
    r1 = llm.chat.completions.create(
        model=MODEL, max_tokens=200,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    r1_text = r1.choices[0].message.content.strip()

    tool_call = extract_tool_call(r1_text) if tool_name != "retrieval_only" else None

    if tool_call:
        # Reject tool calls with null params — model failed to extract a value
        if any(v is None for v in tool_call.get("params", {}).values()):
            return "NOT_FOUND — model could not extract required parameters from context", chunks, None, None
        # Step 3: execute tool
        fn = TOOL_MAP.get(tool_call["tool"])
        if fn:
            try:
                tool_result = fn(**tool_call["params"])
                console.print(f"  [dim]Tool called: {tool_call['tool']}({tool_call['params']})[/dim]")

                # Step 4: second LLM call — synthesize tool result into final answer
                r2 = llm.chat.completions.create(
                    model=MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": "You are a defence analyst. Summarize the tool result in one clear sentence for an officer."},
                        {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}\n\nOriginal question: {question}"}
                    ]
                )
                final = r2.choices[0].message.content.strip()
                return final, chunks, tool_call, tool_result
            except Exception as e:
                return f"Tool error: {e}", chunks, tool_call, None

    return r1_text, chunks, None, None

def print_agent_trace(question, answer, chunks, tool_call, tool_result):
    console.print()
    console.rule("[bold cyan]CHANAKYA AGENT TRACE[/bold cyan]")
    console.print(f"[bold]QUERY[/bold]   : {question}")
    console.print()

    if tool_call:
        console.print(f"[bold yellow]TOOL USED[/bold yellow] : {tool_call['tool']}")
        console.print(f"[yellow]PARAMS[/yellow]    : {tool_call['params']}")
        if tool_result:
            console.print(f"[yellow]RESULT[/yellow]    : {tool_result.get('verdict', tool_result)}")
        console.print()

    console.print(f"[bold green]ANSWER[/bold green]  : {answer}")
    console.print()

    console.print("[dim]Evidence:[/dim]")
    for i, c in enumerate(chunks[:2], 1):
        header = c.payload.get("header", "")
        src = c.payload["source"].split("/")[-1]
        breadcrumb = f"{src} › {header}" if header else src
        console.print(f"  [{i}] [dim]{breadcrumb}[/dim]")
    console.rule()
    console.print()

def main():
    console.print("[bold]Loading documents and building index...[/bold]")
    chunks = load_docs()
    qdrant, vectorizer = build_index(chunks)
    n_docs = len(set(c["source"] for c in chunks))
    console.print(f"Ready: {len(chunks)} chunks from {n_docs} docs | Model: {MODEL}\n")
    console.print("[dim]Agent mode — I can answer questions AND use tools (range, IC compliance, budget, impact time)[/dim]\n")

    llm = get_client()

    while True:
        try:
            question = console.input("[bold cyan]Q:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        answer, chunks, tool_call, tool_result = react_loop(llm, qdrant, vectorizer, question)
        print_agent_trace(question, answer, chunks, tool_call, tool_result)

if __name__ == "__main__":
    main()
