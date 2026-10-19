"""Immutable, startup-built FAQ index shared by requests within a worker."""

import csv
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    """Preserve Unicode words without runtime NLTK corpus downloads."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def read_records(path: Path, required: set[str], allow_empty=frozenset()) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid data columns: {path.name}")
        records = []
        for number, row in enumerate(reader, start=2):
            if any(not isinstance(row.get(key), str) or (key not in allow_empty and not row[key].strip()) for key in required):
                raise ValueError(f"Invalid data row: {path.name}:{number}")
            records.append({key: row[key].strip() for key in required})
    if not records:
        raise ValueError(f"Empty data file: {path.name}")
    return records


class KnowledgeBase:
    def __init__(self, faq_path: Path, navigation_path: Path, threshold: float = 0.3):
        self.faqs = read_records(faq_path, {"Question", "Answer", "Class"})
        self.locations = read_records(navigation_path, {"Building", "Level", "Location", "Instructions"}, {"Instructions"})
        for row in self.locations:
            if not row["Instructions"]:
                row["Instructions"] = "Directions are not available for this location. Please ask hospital staff."
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize, lowercase=False, ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b", sublinear_tf=True,
        )
        # L2-normalized TF-IDF makes the sparse dot product equal cosine similarity.
        # Fit and transform the corpus once, never again in the request path.
        self.matrix = self.vectorizer.fit_transform(row["Question"] for row in self.faqs)
        self._locations = [(normalize(row["Location"]), row) for row in self.locations]
        self._exact_questions = {normalize(row["Question"]) for row in self.faqs}
        self._categories: dict[str, list[dict[str, str]]] = {}
        for row in self.faqs:
            self._categories.setdefault(row["Class"], []).append({
                "Question": row["Question"], "Answer": row["Answer"],
            })

    def categories(self) -> list[str]:
        return list(self._categories)

    def category(self, name: str) -> list[dict[str, str]]:
        return self._categories.get(name, [])

    def navigation(self, query: str) -> list[dict[str, str]]:
        query = normalize(query)
        if not query:
            return []
        # Literal word matching prevents regex injection and punctuation-only matches.
        phrase = f" {query} "
        matches = [(name, row) for name, row in self._locations if phrase in f" {name} "]
        matches.sort(key=lambda item: (item[0] != query, len(item[0])))
        return [row for _, row in matches[:20]]

    def answer(self, message: str) -> dict:
        cleaned = normalize(message)
        result = self._faq_answer(message)
        if cleaned in self._exact_questions:
            return result
        navigation = self.navigation(cleaned)
        direction_query = cleaned.startswith(("where ", "di mana ", "dimanakah ", "lokasi ", "directions ", "how do i get to ", "how to get to "))
        if not navigation and direction_query:
            # Prefer the longest complete location name embedded in a natural question.
            candidates = [(len(name), row) for name, row in self._locations
                          if f" {name} " in f" {cleaned} "]
            if candidates:
                navigation = [max(candidates, key=lambda item: item[0])[1]]
        if navigation:
            row = navigation[0]
            answer = "\n".join(f"{key}: {row[key]}" for key in ("Location", "Building", "Level", "Instructions"))
            return {"response": answer, "suggestions": [row["Location"] for row in navigation[:4]], "source": "navigation"}

        return result

    def _faq_answer(self, message: str) -> dict:
        vector = self.vectorizer.transform([message])
        scores = (self.matrix @ vector.T).toarray().ravel()
        # Stable ordering makes ties repeatable across requests and deployments.
        indices = np.argsort(-scores, kind="stable")[:4]
        suggestions = [self.faqs[index]["Question"] for index in indices if scores[index] > 0]
        best = int(indices[0])
        answer = self.faqs[best]["Answer"] if scores[best] >= self.threshold else None
        return {"response": answer, "suggestions": suggestions, "source": "faq" if answer else "fallback"}
