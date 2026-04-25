try:
    import truststore; truststore.inject_into_ssl()  # corporate SSL proxy (optional)
except ImportError:
    pass
import os
import openai
from rich.console import Console
from ingest import load_docs, build_index, COLLECTION

console = Console()

HF_OLLAMA_ENDPOINT = "https://gtf330-ollama-test.hf.space/v1"
MODEL = os.environ.get("CHANAKYA_MODEL", "tinyllama")

def get_client():
    return openai.OpenAI(api_key="ollama", base_url=HF_OLLAMA_ENDPOINT, timeout=60)

def retrieve(qdrant, vectorizer, query, top_k=4):
    vec = vectorizer.transform([query]).toarray()[0]
    results = qdrant.query_points(collection_name=COLLECTION, query=vec.tolist(), limit=top_k, with_payload=True)
    return results.points

def detect_conflicts(chunks):
    sources = [r.payload["source"] for r in chunks]
    unique_sources = list(dict.fromkeys(sources))
    if len(unique_sources) >= 2:
        return unique_sources
    return []

def ask(llm, chunks, question):
    context = "\n\n".join([
        f"[{r.payload.get('header', r.payload['source'])}] ({r.payload['source']})\n{r.payload['text']}"
        for r in chunks
    ])
    response = llm.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Indian Defence Procurement analyst. "
                    "Answer using ONLY the provided context. "
                    "Be specific — extract exact numbers, names, percentages. "
                    "Cite the source document in your answer. "
                    "If the answer is not in the context, say: NOT_FOUND"
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()

def score_bar(score, width=10):
    filled = int(score * width)
    if score >= 0.7:
        label = "HIGH"
    elif score >= 0.4:
        label = "MED"
    else:
        label = "LOW"
    return f"{'█' * filled}{'░' * (width - filled)} {label} ({score:.2f})"

def print_auditor_trace(results, answer, question):
    console.print()
    console.rule("[bold cyan]CHANAKYA AUDIT TRACE[/bold cyan]")
    console.print(f"[bold]QUERY[/bold]   : {question}")
    console.print()
    console.print(f"[bold green]VERDICT[/bold green] : {answer}")
    console.print()

    conflict_sources = detect_conflicts(results)
    if len(conflict_sources) >= 2:
        console.print(f"[bold red]⚠  CONFLICT ALERT[/bold red] — answer drawn from {len(conflict_sources)} different sources:")
        for s in conflict_sources:
            console.print(f"   [red]•[/red] {s}")
        console.print()

    console.print("[bold yellow]EVIDENCE CHAIN:[/bold yellow]")
    console.rule(style="dim")
    for i, r in enumerate(results, 1):
        header = r.payload.get("header", "")
        source = r.payload["source"]
        score = r.score if hasattr(r, "score") else 0.0
        snippet = r.payload["text"][:200].replace("\n", " ").strip()
        breadcrumb = f"[yellow]{source}[/yellow] › [dim]{header}[/dim]" if header else f"[yellow]{source}[/yellow]"
        console.print(f"[{i}] {breadcrumb}")
        console.print(f"    Score  : {score_bar(score)}")
        console.print(f"    Snippet: [dim]\"{snippet}...\"[/dim]")
        console.print()
    console.rule()
    console.print()

def main():
    print("Loading documents and building index...")
    chunks = load_docs()
    qdrant, vectorizer = build_index(chunks)
    n_docs = len(set(c["source"] for c in chunks))
    print(f"Ready: {len(chunks)} chunks from {n_docs} docs | Model: {MODEL}")
    print("Type a question, or 'quit' to exit.\n")

    llm = get_client()

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        results = retrieve(qdrant, vectorizer, question)
        answer = ask(llm, results, question)
        print_auditor_trace(results, answer, question)

if __name__ == "__main__":
    main()
