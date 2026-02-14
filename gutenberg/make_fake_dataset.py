"""
Create a tiny fake Gutenberg dataset with the same on-disk structure as
`process_data.py` would produce (text/, tokens/, counts/).

It does NOT download anything. It writes a few `data/raw/PG*_raw.txt` files and
then calls `src.pipeline.process_book` to generate:

- data/text/PG*_text.txt
- data/tokens/PG*_tokens.txt   (one token per line)
- data/counts/PG*_counts.txt   (token<TAB>count per line, sorted by frequency)

This is meant for local dev/testing.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re


def _ensure_dirs(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    paths = {
        "raw": data_dir / "raw",
        "text": data_dir / "text",
        "tokens": data_dir / "tokens",
        "counts": data_dir / "counts",
        "metadata": metadata_dir,
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _write_metadata_csv(metadata_dir: Path, rows: list[dict[str, str]]) -> None:
    out_path = metadata_dir / "metadata.csv"
    fieldnames = ["id", "language", "title", "author"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    root = Path(__file__).resolve().parent
    paths = _ensure_dirs(root)

    # A few short "books". Keep them simple so header stripping doesn't erase everything.
    books = [
        {
            "pg": 100,
            "title": "The Index of Fog",
            "author": "Anon",
            "language": "['en']",
            "text": (
                "I.\n"
                "The library was quiet.\n\n"
                "II.\n"
                "A vague light fell across the page, and nothing resolved.\n\n"
                "III.\n"
                "Some sentences were only gestures toward meaning."
            ),
        },
        {
            "pg": 101,
            "title": "Notes on Marble",
            "author": "Anon",
            "language": "['en']",
            "text": (
                "A book can be a room.\n\n"
                "A paragraph can be a door.\n\n"
                "You enter, and the air remembers you."
            ),
        },
        {
            "pg": 102,
            "title": "The Small Rituals",
            "author": "Anon",
            "language": "['en']",
            "text": (
                "Pretend you are searching for love.\n\n"
                "Pretend the corpus is endless.\n\n"
                "Pretend the answer is already highlighted."
            ),
        },
    ]

    # Minimal metadata for `process_data.py` compatibility (it expects id like "PG123").
    _write_metadata_csv(
        paths["metadata"],
        [
            {
                "id": f"PG{b['pg']}",
                "language": b["language"],
                "title": b["title"],
                "author": b["author"],
            }
            for b in books
        ],
    )

    log_file = root / ".log"
    if log_file.exists():
        log_file.unlink()

    for b in books:
        raw_path = paths["raw"] / f"PG{b['pg']}_raw.txt"
        raw_path.write_text(b["text"], encoding="utf-8")

        # Fake "processed" text: for dev, keep it identical to raw.
        clean = b["text"]
        text_path = paths["text"] / f"PG{b['pg']}_text.txt"
        text_path.write_text(clean, encoding="utf-8")

        # Simple, dependency-free tokenizer: lowercase, split on non-letters.
        tokens = [t for t in re.split(r"[^a-zA-Z]+", clean.lower()) if t]
        tokens_path = paths["tokens"] / f"PG{b['pg']}_tokens.txt"
        tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

        counts = Counter(tokens)
        counts_path = paths["counts"] / f"PG{b['pg']}_counts.txt"
        counts_path.write_text(
            "\n".join([f"{w}\t{c}" for w, c in counts.most_common()]) + "\n",
            encoding="utf-8",
        )

        raw_nl = b["text"].count("\n")
        clean_nl = clean.count("\n")
        L = len(tokens)
        V = len(counts)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"PG{b['pg']}\tenglish\t{raw_nl}\t{clean_nl}\t{L}\t{V}\n")

    print("Wrote fake dataset:")
    print(f"- {paths['raw']}")
    print(f"- {paths['text']}")
    print(f"- {paths['tokens']}")
    print(f"- {paths['counts']}")
    print(f"- {paths['metadata'] / 'metadata.csv'}")


if __name__ == "__main__":
    main()
