import json
import sys
from pathlib import Path

def print_report(path):
    with open(path) as f:
        data = json.load(f)

    print(f"{'='*70}")
    print(f"EVAL REPORT")
    print(f"Timestamp  : {data['timestamp']}")
    print(f"Model      : {data['model']}")
    print(f"Embedding  : {data['embedding']}")
    print(f"{'='*70}")
    print(f"Retrieval  : {data['retrieval_accuracy']*100:.0f}%")
    print(f"Answer     : {data['answer_accuracy']*100:.0f}%")
    print(f"{'='*70}\n")

    failures = [r for r in data["results"] if not r["answer_hit"]]
    if failures:
        print(f"ANSWER FAILURES ({len(failures)}):\n")
        for r in failures:
            print(f"[{r['id']}] {r['question']}")
            print(f"  Expected keywords : {r['expected_keywords']}")
            print(f"  Retrieved from    : {r['top_retrieved_source']}")
            print(f"  Answer            : {r['answer'][:300]}")
            print()

    retrieval_failures = [r for r in data["results"] if not r["retrieval_hit"]]
    if retrieval_failures:
        print(f"RETRIEVAL FAILURES ({len(retrieval_failures)}):\n")
        for r in retrieval_failures:
            print(f"[{r['id']}] {r['question']}")
            print(f"  Expected source   : {r['expected_source']}")
            print(f"  Retrieved source  : {r['top_retrieved_source']}")
            print()

    if not failures and not retrieval_failures:
        print("All questions passed.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # pick latest
        results = sorted(Path(".").glob("eval_results_*.json"))
        if not results:
            print("No eval results found.")
            sys.exit(1)
        path = results[-1]
    print_report(path)
