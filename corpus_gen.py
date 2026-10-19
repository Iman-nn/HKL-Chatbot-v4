"""Validate and atomically export the original JSON corpus to a CSV file."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def convert(source: Path, destination: Path) -> int:
    with source.open(encoding="utf-8-sig") as stream:
        corpus = json.load(stream)
    categories = corpus.get("hkl") if isinstance(corpus, dict) else None
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Expected a nonempty hkl category object")
    records = []
    for category, pairs in categories.items():
        if not isinstance(category, str) or not category.strip() or not isinstance(pairs, list):
            raise ValueError("Invalid FAQ category")
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(not isinstance(value, str) or not value.strip() for value in pair)):
                raise ValueError("Each FAQ must contain a nonempty question and answer")
            records.append((pair[0].strip(), pair[1].strip(), category.strip()))
    if not records:
        raise ValueError("No FAQ records found")
    temporary = None
    try:
        # Write in the destination filesystem so replacement is atomic.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(("Question", "Answer", "Class"))
            writer.writerows(records)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "hkl_faq.json")
    parser.add_argument("--output", type=Path, default=ROOT / "hklFAQs.csv")
    arguments = parser.parse_args()
    try:
        count = convert(arguments.source, arguments.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {count} FAQs.")


if __name__ == "__main__":
    main()
