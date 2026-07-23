import json
from src.rag_chain import load_vectorstore, build_rag_chain

def run_eval(test_file="tests/test_cases.json"):
    vectorstore = load_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    with open(test_file) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        answer = chain.invoke(case["question"])
        hit = any(kw.lower() in answer.lower() for kw in case["expected_keywords"]) if case["expected_keywords"] else None
        results.append({
            "question": case["question"],
            "category": case["category"],
            "answer": answer,
            "keyword_hit": hit,
        })

    scored = [r for r in results if r["keyword_hit"] is not None]
    accuracy = sum(r["keyword_hit"] for r in scored) / len(scored) if scored else 0
    print(f"Keyword-match accuracy: {accuracy:.0%} ({len(scored)} scorable cases)")
    for r in results:
        print(f"\n[{r['category']}] Q: {r['question']}\nA: {r['answer'][:200]}")
    return results

if __name__ == "__main__":
    run_eval()