"""Local microbenchmark of cached vs repeated corpus transforms, not a load test."""

import json
from pathlib import Path
import statistics
import time

from hkl_chatbot import ROOT
from hkl_chatbot.knowledge import KnowledgeBase


def main():
    knowledge = KnowledgeBase(ROOT / "hklFAQsv2.csv", ROOT / "hklNavigation.csv")
    queries = [row["Question"] for row in knowledge.faqs]
    cached, recomputed = [], []
    # Warm the vectorizer and sparse matrix paths before timing.
    knowledge.answer(queries[0])
    for index in range(100):
        question = queries[index % len(queries)]
        start = time.perf_counter()
        vector = knowledge.vectorizer.transform([question])
        _ = (knowledge.matrix @ vector.T).toarray()
        cached.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        vector = knowledge.vectorizer.transform([question])
        corpus = knowledge.vectorizer.transform(queries)
        _ = (corpus @ vector.T).toarray()
        recomputed.append((time.perf_counter() - start) * 1000)
    report = {
        "faq_rows": len(knowledge.faqs), "navigation_rows": len(knowledge.locations),
        "samples": 100, "cached_median_ms": round(statistics.median(cached), 3),
        "recomputed_median_ms": round(statistics.median(recomputed), 3),
        "median_ratio": round(statistics.median(recomputed) / statistics.median(cached), 2),
        "scope": "Retrieval only, same TF-IDF preprocessing; not original NLTK pipeline or end-to-end load.",
    }
    output = ROOT / "artifacts" / "retrieval-benchmark.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
