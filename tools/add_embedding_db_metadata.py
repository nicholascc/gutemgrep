from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path
from typing import Iterable


def _load_metadata(metadata_csv: Path) -> tuple[dict[int, str], dict[int, str], dict[int, int], dict[int, str]]:
    if not metadata_csv.exists():
        return {}, {}, {}, {}
    titles: dict[int, str] = {}
    authors: dict[int, str] = {}
    downloads: dict[int, int] = {}
    languages: dict[int, str] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            raw_id = (row.get("id") or "").strip()
            if not raw_id.startswith("PG"):
                continue
            try:
                book_id = int(raw_id[2:])
            except Exception:
                continue
            title = (row.get("title") or "").strip()
            if title:
                titles[book_id] = title
            author = (row.get("author") or "").strip()
            if author:
                authors[book_id] = author
            downloads_raw = (row.get("downloads") or "").strip()
            if downloads_raw:
                try:
                    downloads[book_id] = int(float(downloads_raw))
                except Exception:
                    pass
            language = (row.get("language") or "").strip()
            if language:
                languages[book_id] = language
    return titles, authors, downloads, languages


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS book_metadata (
          book_id INTEGER PRIMARY KEY,
          book_title TEXT,
          author TEXT,
          language TEXT,
          downloads INTEGER
        )
        """
    )
    conn.commit()


def _iter_book_ids(conn: sqlite3.Connection, *, table: str) -> Iterable[int]:
    rows = conn.execute(
        f"SELECT DISTINCT book_id FROM {table} WHERE book_id IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            yield int(row[0])
        except Exception:
            continue


def _upsert_book_metadata(
    conn: sqlite3.Connection,
    *,
    book_ids: Iterable[int],
    titles: dict[int, str],
    authors: dict[int, str],
    downloads: dict[int, int],
    languages: dict[int, str],
    only_missing: bool,
) -> int:
    rows = []
    for book_id in book_ids:
        rows.append(
            (
                int(book_id),
                titles.get(book_id),
                authors.get(book_id),
                languages.get(book_id),
                int(downloads.get(book_id)) if book_id in downloads else None,
            )
        )
    if not rows:
        return 0
    if only_missing:
        conn.executemany(
            """
            INSERT OR IGNORE INTO book_metadata (
              book_id,
              book_title,
              author,
              language,
              downloads
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    else:
        conn.executemany(
            """
            INSERT OR REPLACE INTO book_metadata (
              book_id,
              book_title,
              author,
              language,
              downloads
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add book metadata to embedding.db.")
    parser.add_argument("--db", default="embedding.db", help="Path to embedding.db")
    parser.add_argument(
        "--metadata-csv", default="gutenberg/metadata/metadata.csv", help="Metadata CSV path"
    )
    parser.add_argument("--table", default=os.environ.get("EMBEDDINGS_TABLE", "embeddings"))
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only insert missing book_id rows (no updates).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    db_path = (root / args.db).resolve()
    metadata_csv = (root / args.metadata_csv).resolve()

    if not db_path.exists():
        raise SystemExit(f"Missing db: {db_path}")

    titles, authors, downloads, languages = _load_metadata(metadata_csv)
    if not titles and not authors and not downloads and not languages:
        raise SystemExit(f"No metadata loaded from {metadata_csv}")

    with sqlite3.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        book_ids = list(_iter_book_ids(conn, table=args.table))
        total = _upsert_book_metadata(
            conn,
            book_ids=book_ids,
            titles=titles,
            authors=authors,
            downloads=downloads,
            languages=languages,
            only_missing=bool(args.only_missing),
        )

    print(f"OK: upserted {total} book_metadata rows into {db_path}")


if __name__ == "__main__":
    main()
