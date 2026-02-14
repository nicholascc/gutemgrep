from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests


@dataclass(frozen=True)
class ParagraphRow:
    embed_id: int
    paragraph_text: str
    prev_embed_id: Optional[int]
    next_embed_id: Optional[int]
    book_title: Optional[str]
    book_id: Optional[int]


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _load_titles(metadata_csv: Path) -> dict[int, str]:
    if not metadata_csv.exists():
        return {}
    by_book_id: dict[int, str] = {}
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
                by_book_id[book_id] = title
    return by_book_id


def _embed_batch(*, api_key: str, model: str, texts: list[str]) -> list[np.ndarray]:
    if any((not isinstance(t, str)) or (not t.strip()) for t in texts):
        raise ValueError("Embedding batch contains empty text.")

    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": texts, "encoding_format": "float"},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI embeddings error ({resp.status_code}): {resp.text}")

    data = resp.json()
    items = data.get("data") or []
    if not isinstance(items, list) or len(items) != len(texts):
        raise RuntimeError(f"Unexpected embeddings response shape: {data}")

    # The API includes an index per item; sort to be safe.
    items_sorted = sorted(items, key=lambda x: int(x.get("index", 0)))
    out: list[np.ndarray] = []
    for item in items_sorted:
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Missing embedding in response.")
        out.append(np.asarray(embedding, dtype=np.float32))
    return out


def _retry_embed_batch(
    *,
    api_key: str,
    model: str,
    texts: list[str],
    max_retries: int = 6,
) -> list[np.ndarray]:
    delay_s = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _embed_batch(api_key=api_key, model=model, texts=texts)
        except Exception as e:
            last_err = e
            msg = str(e)
            retriable = any(code in msg for code in ["(429)", "(500)", "(502)", "(503)", "(504)"])
            if attempt >= max_retries or not retriable:
                raise
            time.sleep(delay_s)
            delay_s = min(20.0, delay_s * 2.0)
    assert last_err is not None
    raise last_err


def _iter_text_files(text_dir: Path) -> Iterable[tuple[int, Path]]:
    for path in sorted(text_dir.glob("PG*_text.txt")):
        m = re.match(r"PG(\d+)_text\.txt$", path.name)
        if not m:
            continue
        yield int(m.group(1)), path


def _create_schema(conn: sqlite3.Connection, *, table: str, overwrite: bool) -> None:
    if overwrite:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          embed_id INTEGER PRIMARY KEY,
          paragraph_text TEXT NOT NULL,
          prev_embed_id INTEGER,
          next_embed_id INTEGER,
          embedding BLOB NOT NULL,
          book_title TEXT,
          book_id INTEGER
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_book_id ON {table}(book_id)")
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding.db from Gutenberg text/ files.")
    parser.add_argument("--text-dir", default="gutenberg/data/text", help="Directory of PG*_text.txt files.")
    parser.add_argument("--metadata-csv", default="gutenberg/metadata/metadata.csv", help="Metadata CSV with titles.")
    parser.add_argument("--out-db", default="embedding.db", help="Output SQLite path.")
    parser.add_argument("--table", default=os.environ.get("EMBEDDINGS_TABLE", "embeddings"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-paragraphs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Drop and recreate the embeddings table.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY.")

    root = Path(__file__).resolve().parents[1]
    text_dir = (root / args.text_dir).resolve()
    metadata_csv = (root / args.metadata_csv).resolve()
    out_db = (root / args.out_db).resolve()

    if not text_dir.exists():
        raise SystemExit(f"Missing text dir: {text_dir}")

    titles = _load_titles(metadata_csv)

    rows: list[ParagraphRow] = []
    next_embed_id = 1
    for book_id, path in _iter_text_files(text_dir):
        title = titles.get(book_id)
        text = path.read_text(encoding="utf-8", errors="replace")
        paras_all = _split_paragraphs(text)
        paras = [p for p in paras_all if len(p) >= int(args.min_chars)]
        if not paras:
            continue

        start = next_embed_id
        end = start + len(paras) - 1
        for i, para in enumerate(paras):
            embed_id = start + i
            prev_id = embed_id - 1 if embed_id > start else None
            next_id = embed_id + 1 if embed_id < end else None
            rows.append(
                ParagraphRow(
                    embed_id=embed_id,
                    paragraph_text=para,
                    prev_embed_id=prev_id,
                    next_embed_id=next_id,
                    book_title=title,
                    book_id=book_id,
                )
            )

        next_embed_id = end + 1
        if args.max_paragraphs and len(rows) >= int(args.max_paragraphs):
            rows = rows[: int(args.max_paragraphs)]
            break

    if not rows:
        raise SystemExit(f"No paragraphs found in {text_dir}.")

    out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(out_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _create_schema(conn, table=args.table, overwrite=bool(args.overwrite))

        inserted = 0
        for i in range(0, len(rows), int(args.batch_size)):
            batch = rows[i : i + int(args.batch_size)]
            texts = [r.paragraph_text for r in batch]
            vecs = _retry_embed_batch(api_key=api_key, model=args.model, texts=texts)

            payload = []
            for r, v in zip(batch, vecs, strict=True):
                payload.append(
                    (
                        int(r.embed_id),
                        r.paragraph_text,
                        int(r.prev_embed_id) if r.prev_embed_id is not None else None,
                        int(r.next_embed_id) if r.next_embed_id is not None else None,
                        sqlite3.Binary(v.astype(np.float32).tobytes()),
                        r.book_title,
                        int(r.book_id) if r.book_id is not None else None,
                    )
                )

            conn.executemany(
                f"""
                INSERT OR REPLACE INTO {args.table} (
                  embed_id,
                  paragraph_text,
                  prev_embed_id,
                  next_embed_id,
                  embedding,
                  book_title,
                  book_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()
            inserted += len(batch)
            print(f"Embedded {inserted}/{len(rows)} paragraphs", end="\r")

    print()
    print(f"OK: wrote {inserted} rows to {out_db} (table: {args.table}, model: {args.model})")


if __name__ == "__main__":
    main()

