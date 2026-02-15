from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import requests

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class ParagraphRow:
    embed_id: int
    paragraph_text: str
    prev_embed_id: Optional[int]
    next_embed_id: Optional[int]
    book_title: Optional[str]
    book_id: Optional[int]

_thread_local = threading.local()


def _get_requests_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def _iter_paragraphs_from_file(path: Path, *, min_tokens: int, min_chars: int) -> Iterator[str]:
    """
    Chunking algorithm:
    - Treat "paragraphs" as blocks separated by blank lines (i.e., double-newlines in the source).
    - Build a chunk by concatenating paragraphs until the chunk has at least `min_tokens` tokens.
    - Once the threshold is reached, end the chunk at the next paragraph boundary.
    """
    chunk_paras: list[str] = []
    chunk_tokens = 0
    prev_chunk: str | None = None

    para_lines: list[str] = []

    def flush_paragraph() -> Optional[str]:
        if not para_lines:
            return None
        # Processed Gutenberg text is often fixed-width; unwrap wrapped lines.
        para = " ".join(line.strip() for line in para_lines).strip()
        para_lines.clear()
        return para or None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                para = flush_paragraph()
                if para is None:
                    continue

                chunk_paras.append(para)
                chunk_tokens += _count_tokens(para)

                if chunk_tokens >= min_tokens:
                    chunk = "\n\n".join(chunk_paras).strip()
                    chunk_paras.clear()
                    chunk_tokens = 0

                    if chunk and len(chunk) >= min_chars:
                        if prev_chunk is not None:
                            yield prev_chunk
                        prev_chunk = chunk
                continue

            para_lines.append(line)

    # Flush final paragraph (EOF might not end with a blank line).
    para = flush_paragraph()
    if para is not None:
        chunk_paras.append(para)
        chunk_tokens += _count_tokens(para)

    tail = "\n\n".join(chunk_paras).strip()
    if tail and len(tail) >= min_chars:
        if prev_chunk is not None:
            prev_chunk = (prev_chunk + "\n\n" + tail).strip()
        else:
            prev_chunk = tail

    if prev_chunk is not None:
        yield prev_chunk


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

    resp = _get_requests_session().post(
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


def _embed_and_insert(
    conn: sqlite3.Connection,
    *,
    table: str,
    api_key: str,
    model: str,
    rows: list[ParagraphRow],
) -> None:
    payload = _embed_payload(api_key=api_key, model=model, rows=rows)
    _insert_payload(conn, table=table, payload=payload)


def _embed_payload(*, api_key: str, model: str, rows: list[ParagraphRow]) -> list[tuple]:
    texts = [r.paragraph_text for r in rows]
    vecs = _retry_embed_batch(api_key=api_key, model=model, texts=texts)

    payload = []
    for r, v in zip(rows, vecs, strict=True):
        payload.append(
            (
                int(r.embed_id),
                r.paragraph_text,
                int(r.prev_embed_id) if r.prev_embed_id is not None else None,
                int(r.next_embed_id) if r.next_embed_id is not None else None,
                v.astype(np.float32).tobytes(),
                r.book_title,
                int(r.book_id) if r.book_id is not None else None,
            )
        )
    return payload


def _insert_payload(conn: sqlite3.Connection, *, table: str, payload: list[tuple]) -> None:
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding.db from Gutenberg text/ files.")
    parser.add_argument("--text-dir", default="gutenberg/data/text", help="Directory of PG*_text.txt files.")
    parser.add_argument("--metadata-csv", default="gutenberg/metadata/metadata.csv", help="Metadata CSV with titles.")
    parser.add_argument("--out-db", default="embedding.db", help="Output SQLite path.")
    parser.add_argument("--table", default=os.environ.get("EMBEDDINGS_TABLE", "embeddings"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("EMBED_CONCURRENCY", "1")),
        help="Number of concurrent embedding requests.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=0,
        help="Max embedding batches in flight; 0 defaults to 2*concurrency.",
    )
    parser.add_argument("--max-books", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-tokens", type=int, default=50)
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

    out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(out_db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _create_schema(conn, table=args.table, overwrite=bool(args.overwrite))

        batch_size = max(1, int(args.batch_size))
        concurrency = max(1, int(args.concurrency))
        max_inflight = int(args.max_inflight)
        if max_inflight <= 0:
            max_inflight = max(1, 2 * concurrency)
        min_tokens = max(1, int(args.min_tokens))
        min_chars = max(1, int(args.min_chars))
        max_paragraphs = max(0, int(args.max_paragraphs))

        max_books = max(0, int(args.max_books))
        text_files = list(_iter_text_files(text_dir))
        if max_books:
            text_files = text_files[:max_books]
        total_books = len(text_files)
        if total_books == 0:
            raise SystemExit(f"No PG*_text.txt files found in {text_dir}.")

        batch: list[ParagraphRow] = []
        inserted = 0
        seen = 0
        next_embed_id = 1

        # Embed + insert: optionally in parallel (embedding requests only; SQLite writes stay single-threaded).
        def insert_one_payload(payload: list[tuple]) -> None:
            nonlocal inserted
            _insert_payload(conn, table=args.table, payload=payload)
            conn.commit()
            inserted += len(payload)
            print(f"Books {books_done}/{total_books} | Embedded {inserted} paragraphs", end="\r")

        inflight: set[concurrent.futures.Future[list[tuple]]] = set()
        books_done = 0

        def drain_one(*, pool: concurrent.futures.Executor) -> None:
            done, _ = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                inflight.remove(fut)
                insert_one_payload(fut.result())

        def drain_all(*, pool: concurrent.futures.Executor) -> None:
            for fut in concurrent.futures.as_completed(list(inflight)):
                inflight.remove(fut)
                insert_one_payload(fut.result())

        def submit_batch(*, pool: concurrent.futures.Executor | None, rows: list[ParagraphRow]) -> None:
            if not rows:
                return
            if pool is None:
                insert_one_payload(_embed_payload(api_key=api_key, model=args.model, rows=rows))
                return
            inflight.add(pool.submit(_embed_payload, api_key=api_key, model=args.model, rows=rows))
            if len(inflight) >= max_inflight:
                drain_one(pool=pool)

        pool: concurrent.futures.Executor | None
        if concurrency <= 1:
            pool = None
        else:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

        try:
            for book_id, path in text_files:
                title = titles.get(book_id)

                pending: ParagraphRow | None = None
                prev_embed_id: int | None = None

                for para in _iter_paragraphs_from_file(path, min_tokens=min_tokens, min_chars=min_chars):
                    if max_paragraphs and seen >= max_paragraphs:
                        break
                    seen += 1

                    embed_id = next_embed_id
                    next_embed_id += 1

                    if pending is not None:
                        batch.append(
                            ParagraphRow(
                                embed_id=pending.embed_id,
                                paragraph_text=pending.paragraph_text,
                                prev_embed_id=pending.prev_embed_id,
                                next_embed_id=embed_id,
                                book_title=pending.book_title,
                                book_id=pending.book_id,
                            )
                        )
                        if len(batch) >= batch_size:
                            submit_batch(pool=pool, rows=batch)
                            batch = []

                    pending = ParagraphRow(
                        embed_id=embed_id,
                        paragraph_text=para,
                        prev_embed_id=prev_embed_id,
                        next_embed_id=None,
                        book_title=title,
                        book_id=book_id,
                    )
                    prev_embed_id = embed_id

                if pending is not None and (not max_paragraphs or seen <= max_paragraphs):
                    batch.append(
                        ParagraphRow(
                            embed_id=pending.embed_id,
                            paragraph_text=pending.paragraph_text,
                            prev_embed_id=pending.prev_embed_id,
                            next_embed_id=None,
                            book_title=pending.book_title,
                            book_id=pending.book_id,
                        )
                    )
                    if len(batch) >= batch_size:
                        submit_batch(pool=pool, rows=batch)
                        batch = []

                books_done += 1
                print(f"Books {books_done}/{total_books} | Embedded {inserted} paragraphs", end="\r")

                if max_paragraphs and seen >= max_paragraphs:
                    break

            if batch:
                submit_batch(pool=pool, rows=batch)
                batch = []

            if pool is not None:
                drain_all(pool=pool)
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

        if inserted == 0:
            raise SystemExit(f"No paragraphs found in {text_dir}.")

    print()
    print(f"OK: wrote {inserted} rows to {out_db} (table: {args.table}, model: {args.model})")


if __name__ == "__main__":
    main()
