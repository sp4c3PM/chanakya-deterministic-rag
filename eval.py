try:
    import truststore; truststore.inject_into_ssl()  # corporate SSL proxy (optional)
except ImportError:
    pass
import os
import time
import subprocess
import json
from datetime import datetime
from ingest import load_docs, build_index, COLLECTION
from eval_set import EVAL_SET
import openai

HF_OLLAMA_ENDPOINT = "https://gtf330-ollama-test.hf.space/v1"
MODEL = os.environ.get("CHANAKYA_MODEL", "tinyllama")

def get_client():
    return openai.OpenAI(api_key="ollama", base_url=HF_OLLAMA_ENDPOINT, timeout=60)

def retrieve(client, vectorizer, query, top_k=4):
    vec = vectorizer.transform([query]).toarray()[0]
    return client.query_points(collection_name=COLLECTION, query=vec.tolist(), limit=top_k).points

def ask_llm(context_chunks, question):
    client = get_client()
    context = "\n\n".join([f"[{r.payload['source']}]\n{r.payload['text']}" for r in context_chunks])
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": "You are an Indian Defence Procurement auditor. Use ONLY the provided context. Extract the specific percentage, value, or name asked for in one sentence. Cite the source document. If not found, say: NOT_FOUND"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

def score_retrieval(results, expected_source):
    sources = [r.payload["source"] for r in results]
    return any(expected_source == s for s in sources)

def score_answer(answer, keywords):
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)

def run_eval(client, vectorizer):
    results_log = []
    retrieval_hits = 0
    answer_hits = 0

    print(f"Running eval on {len(EVAL_SET)} questions...\n")
    print(f"{'ID':<5} {'Retrieval':<12} {'Answer':<10} Question")
    print("-" * 80)

    for item in EVAL_SET:
        retrieved = retrieve(client, vectorizer, item["question"])
        answer = ask_llm(retrieved, item["question"])
        time.sleep(0.5)

        r_hit = score_retrieval(retrieved, item["expected_source"])
        a_hit = score_answer(answer, item["expected_keywords"])
        retrieval_hits += int(r_hit)
        answer_hits += int(a_hit)

        print(f"{item['id']:<5} {'✓' if r_hit else '✗':<12} {'✓' if a_hit else '✗':<10} {item['question'][:60]}")

        results_log.append({
            "id": item["id"],
            "question": item["question"],
            "expected_source": item["expected_source"],
            "top_retrieved_source": retrieved[0].payload["source"],
            "retrieval_hit": r_hit,
            "answer": answer,
            "expected_keywords": item["expected_keywords"],
            "answer_hit": a_hit,
        })

    n = len(EVAL_SET)
    print("-" * 80)
    print(f"\nRetrieval accuracy : {retrieval_hits}/{n} = {retrieval_hits/n*100:.0f}%")
    print(f"Answer accuracy    : {answer_hits}/{n} = {answer_hits/n*100:.0f}%")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval_results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "model": MODEL,
            "embedding": "tfidf",
            "n_chunks": None,
            "retrieval_accuracy": retrieval_hits / n,
            "answer_accuracy": answer_hits / n,
            "results": results_log,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    chunks = load_docs()
    client, vectorizer = build_index(chunks)
    run_eval(client, vectorizer)
