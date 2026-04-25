try:
    import truststore; truststore.inject_into_ssl()  # corporate SSL proxy (optional)
except ImportError:
    pass
import os
import json
import openai
from rich.console import Console
from rich.table import Table
from rich import box
from ingest import load_docs, build_index, COLLECTION
from overrides import get_override, save_override, list_overrides

HF_OLLAMA_ENDPOINT = "https://gtf330-ollama-test.hf.space/v1"
MODEL = os.environ.get("CHANAKYA_MODEL", "tinyllama")

console = Console()

ATOM_SCHEMA = {
    "capability":           "Speed and range (e.g. Mach 2.8, 290-450 km)",
    "launch_platform":      "Air / Land / Sea / Multi (how it is deployed)",
    "indigenous_content":   "IC% by value (e.g. 50%, 65%)",
    "procurement_category": "DAP category (e.g. Buy Indian-IDDM, Buy Global)",
    "unit_cost":            "Cost per unit or total contract value (e.g. Rs 30 crore)",
    "context":              "WHERE this value applies (e.g. Philippines export 2022, domestic production estimate)",
}

def get_client():
    return openai.OpenAI(api_key="ollama", base_url=HF_OLLAMA_ENDPOINT, timeout=60)

def retrieve_for_entity(qdrant, vectorizer, entity, top_k=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = vectorizer.transform([entity]).toarray()[0]
    return qdrant.query_points(
        collection_name=COLLECTION,
        query=vec.tolist(),
        limit=top_k,
        with_payload=True
    ).points

def extract_fields_from_chunk(llm, chunk_text, source, entity):
    prompt = f"""Extract facts about "{entity}" from the text below.

Rules:
- Output ONLY a JSON object with exactly these keys: capability, launch_platform, indigenous_content, procurement_category, unit_cost
- Each value must be a SHORT factual string copied from the text (e.g. "Mach 2.8, 290-450 km" or "Rs 375 million")
- If the text does not mention a field, output exactly: NOT_SPECIFIED
- Do NOT copy field descriptions. Do NOT guess or infer.

Example output:
{{"capability": "Mach 2.8, range 290-450 km", "launch_platform": "Air, Land, Sea", "indigenous_content": "65%", "procurement_category": "NOT_SPECIFIED", "unit_cost": "Rs 375 million", "context": "Philippines export contract 2022"}}

Text:
{chunk_text[:800]}

JSON:"""

    try:
        response = llm.chat.completions.create(
            model=MODEL,
            max_tokens=150,
            messages=[
                {"role": "system", "content": "Output valid JSON only. No explanation. No markdown."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        # Reject values that look like field descriptions (> 80 chars or no actual data)
        for k in ATOM_SCHEMA:
            v = parsed.get(k, "NOT_SPECIFIED")
            if len(str(v)) > 80 or str(v).lower() in ("null", "none", "n/a", ""):
                parsed[k] = "NOT_SPECIFIED"
        return parsed
    except Exception:
        return {k: "NOT_SPECIFIED" for k in ATOM_SCHEMA}

def merge_extractions(extractions):
    merged = {k: [] for k in ATOM_SCHEMA}
    for source, fields in extractions:
        for k, v in fields.items():
            if v and v not in ("NOT_SPECIFIED", "PARSE_ERROR", ""):
                merged[k].append({"value": v, "source": source})
    return merged

def detect_field_conflicts(merged):
    conflicts = {}
    for field, entries in merged.items():
        unique_vals = list({e["value"] for e in entries})
        if len(unique_vals) > 1:
            conflicts[field] = entries
    return conflicts

def apply_overrides(entity, merged, conflicts):
    applied = {}
    for field in list(conflicts.keys()):
        override = get_override(entity, field)
        if override:
            merged[field] = [{"value": override["value"], "source": f"[OVERRIDE by {override['override_by']} on {override['override_date']}]"}]
            del conflicts[field]
            applied[field] = override
    return applied

def prompt_overrides(entity, conflicts):
    if not conflicts:
        return
    console.print("\n[bold yellow]Resolve conflicts? (press Enter to skip each)[/bold yellow]")
    for field, entries in conflicts.items():
        console.print(f"\n  [yellow]{field}[/yellow] — {len(entries)} conflicting values:")
        for i, e in enumerate(entries, 1):
            console.print(f"    [{i}] {e['value']}  ←  {e['source'].split('/')[-1]}")
        console.print(f"    [0] Enter custom value")
        choice = console.input("  Choose [1/2/0/Enter to skip]: ").strip()
        if not choice:
            continue
        if choice == "0":
            val = console.input("  Value: ").strip()
            ctx = console.input("  Context (e.g. 'domestic procurement'): ").strip()
        elif choice.isdigit() and 1 <= int(choice) <= len(entries):
            val = entries[int(choice)-1]["value"]
            ctx = console.input("  Context for this value: ").strip()
        else:
            continue
        analyst = console.input("  Your name/role: ").strip() or "analyst"
        save_override(entity, field, val, ctx, analyst)
        console.print(f"  [green]✓ Override saved for {field}[/green]")

def print_factsheet(entity, merged, conflicts, sources_used):
    console.print()
    console.rule(f"[bold cyan]FACT SHEET — {entity.upper()}[/bold cyan]")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold yellow")
    table.add_column("Field", style="yellow", width=22)
    table.add_column("Value", width=30)
    table.add_column("Source", style="dim", width=30)

    for field in ATOM_SCHEMA:
        entries = merged[field]
        if not entries:
            table.add_row(field, "[dim]NOT_SPECIFIED[/dim]", "—")
        elif field in conflicts:
            vals = " | ".join(e["value"] for e in entries)
            srcs = " | ".join(e["source"].split("/")[-1] for e in entries)
            table.add_row(field, f"[red]{vals}[/red]", srcs)
        else:
            table.add_row(field, f"[green]{entries[0]['value']}[/green]",
                          entries[0]["source"].split("/")[-1])

    console.print(table)

    if conflicts:
        console.print("[bold red]⚠  CONFLICTS DETECTED[/bold red]")
        for field, entries in conflicts.items():
            console.print(f"   [yellow]{field}[/yellow]:")
            for e in entries:
                console.print(f"     • {e['value']}  ←  {e['source'].split('/')[-1]}")
        console.print()

    # Gap Analysis — fields empty across ALL docs
    gaps = [f for f in ATOM_SCHEMA if not merged[f]]
    if gaps:
        console.print("[bold yellow]⚠  GAP ANALYSIS — not found in any source:[/bold yellow]")
        for g in gaps:
            console.print(f"   [dim]• {g} — NOT IN CORPUS[/dim]")
        console.print()

    console.print(f"[dim]Sources searched: {', '.join(s.split('/')[-1] for s in sources_used)}[/dim]")
    console.rule()
    console.print()

def build_factsheet(entity, qdrant, vectorizer, llm):
    console.print(f"\n[cyan]Retrieving chunks for:[/cyan] {entity}...")
    chunks = retrieve_for_entity(qdrant, vectorizer, entity)

    # Group by source — extract per doc, not per chunk
    by_source = {}
    for c in chunks:
        src = c.payload["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(c.payload["text"])

    extractions = []
    for source, texts in by_source.items():
        combined = " ".join(texts)[:1200]
        console.print(f"  [dim]Extracting from {source.split('/')[-1]}...[/dim]")
        fields = extract_fields_from_chunk(llm, combined, source, entity)
        extractions.append((source, fields))

    merged = merge_extractions(extractions)
    conflicts = detect_field_conflicts(merged)

    # Apply any existing expert overrides
    applied = apply_overrides(entity, merged, conflicts)
    if applied:
        console.print(f"  [green]Applied {len(applied)} expert override(s): {', '.join(applied.keys())}[/green]")

    print_factsheet(entity, merged, conflicts, list(by_source.keys()))

    # Prompt user to resolve remaining conflicts
    if conflicts:
        prompt_overrides(entity, conflicts)

def main():
    console.print("[bold]Loading documents and building index...[/bold]")
    chunks = load_docs()
    qdrant, vectorizer = build_index(chunks)
    n_docs = len(set(c["source"] for c in chunks))
    console.print(f"Ready: {len(chunks)} chunks from {n_docs} docs | Model: {MODEL}\n")

    llm = get_client()

    while True:
        try:
            entity = console.input("[bold cyan]Entity:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break

        if not entity or entity.lower() in ("quit", "exit", "q"):
            break

        build_factsheet(entity, qdrant, vectorizer, llm)

if __name__ == "__main__":
    main()
