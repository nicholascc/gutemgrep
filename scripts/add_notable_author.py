#!/usr/bin/env python3
"""
Add a notable_author column to a Gutenberg-style metadata CSV based on a list of notable authors.
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


IGNORE_PAREN_WORDS = {
    "omit",
    "english",
    "corpus",
    "translations",
    "translation",
    "strictly",
    "original",
    "pre",
    "post",
    "work",
    "limited",
    "include",
    "included",
    "flag",
    "via",
    "want",
    "only",
    "if",
    "you",
    "for",
    "in",
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
}
SUFFIX_TOKENS = {"jr", "sr", "iii", "iv", "ii"}


def _strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _normalize(text: str) -> str:
    text = _strip_diacritics(text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_variants(text: str) -> set[str]:
    norm = _normalize(text)
    variants = {norm} if norm else set()
    tokens = norm.split()
    if tokens and tokens[-1] in SUFFIX_TOKENS:
        trimmed = " ".join(tokens[:-1])
        if trimmed:
            variants.add(trimmed)
    return variants


def _maybe_alias_from_paren(paren: str) -> str | None:
    candidate = re.split(r"[;]", paren, maxsplit=1)[0].strip()
    if not candidate:
        return None
    tokens = set(_normalize(candidate).split())
    if tokens & IGNORE_PAREN_WORDS:
        return None
    return candidate


def _load_notable_names(path: Path) -> tuple[set[str], set[str]]:
    notable = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", line)
        if m:
            base = m.group(1).strip()
            paren = m.group(2).strip()
            if base:
                notable.update(_normalize_variants(base))
            alias = _maybe_alias_from_paren(paren)
            if alias:
                notable.update(_normalize_variants(alias))
        else:
            notable.update(_normalize_variants(line))
    single_tokens = {name for name in notable if " " not in name}
    return notable, single_tokens


def _author_variants(author: str) -> set[str]:
    if not author:
        return set()
    variants = set()
    variants.update(_normalize_variants(author))
    if "," in author:
        parts = [p.strip() for p in author.split(",", 1)]
        if len(parts) == 2:
            last, rest = parts[0], parts[1]
            if rest:
                reordered = f"{rest} {last}".strip()
                variants.update(_normalize_variants(reordered))
    return variants


def _split_authors(author: str) -> list[str]:
    # Keep conservative splitting to avoid breaking institutional names.
    if ";" in author:
        return [a.strip() for a in author.split(";") if a.strip()]
    if "|" in author:
        return [a.strip() for a in author.split("|") if a.strip()]
    return [author]


def _is_notable(author: str, notable: set[str], single_tokens: set[str]) -> bool:
    for part in _split_authors(author):
        variants = _author_variants(part)
        if any(v in notable for v in variants):
            return True
        for v in variants:
            tokens = v.split()
            if not tokens:
                continue
            last = tokens[-1]
            if last in single_tokens:
                return True
            if v in single_tokens:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add a notable_author column to a metadata CSV."
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Path to input metadata CSV.",
    )
    parser.add_argument(
        "--authors-list",
        required=True,
        help="Path to notable authors list (one per line).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to write the output CSV.",
    )
    parser.add_argument(
        "--column-name",
        default="notable_author",
        help="Name of the output column (default: notable_author).",
    )
    parser.add_argument(
        "--replace-column",
        action="store_true",
        help="Replace column if it already exists.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_csv)
    authors_list_path = Path(args.authors_list)
    output_path = Path(args.output_csv)

    notable, single_tokens = _load_notable_names(authors_list_path)

    with metadata_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if args.column_name in fieldnames and not args.replace_column:
            raise SystemExit(
                f"Column already exists: {args.column_name}. Use --replace-column to overwrite."
            )
        if args.column_name not in fieldnames:
            fieldnames.append(args.column_name)

        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                author = row.get("author", "")
                row[args.column_name] = "1" if _is_notable(author, notable, single_tokens) else "0"
                writer.writerow(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
