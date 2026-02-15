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


@dataclass(frozen=True)
class EmbedResult:
    payload: list[tuple]
    total_tokens: int
    embed_s: float

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


def _load_metadata(metadata_csv: Path) -> tuple[dict[int, str], dict[int, int]]:
    if not metadata_csv.exists():
        return {}, {}
    by_book_id: dict[int, str] = {}
    downloads_by_id: dict[int, int] = {}
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
            downloads_raw = (row.get("downloads") or "").strip()
            if downloads_raw:
                try:
                    downloads_by_id[book_id] = int(float(downloads_raw))
                except Exception:
                    continue
    return by_book_id, downloads_by_id


def _embed_batch(*, api_key: str, model: str, task: str, texts: list[str]) -> tuple[list[np.ndarray], int]:
    if any((not isinstance(t, str)) or (not t.strip()) for t in texts):
        raise ValueError("Embedding batch contains empty text.")

    resp = _get_requests_session().post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "task": task, "input": texts, "truncate": True},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Jina embeddings error ({resp.status_code}): {resp.text}")

    data = resp.json()
    total_tokens_int = 0
    usage = data.get("usage") or {}
    if isinstance(usage, dict) and "total_tokens" in usage:
        try:
            total_tokens_int = int(usage["total_tokens"])
        except Exception:
            total_tokens_int = 0
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
    return out, total_tokens_int


def _retry_embed_batch(
    *,
    api_key: str,
    model: str,
    task: str,
    texts: list[str],
    max_retries: int = 6,
) -> tuple[list[np.ndarray], int]:
    delay_s = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return _embed_batch(api_key=api_key, model=model, task=task, texts=texts)
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
        conn.execute("DROP TABLE IF EXISTS embedded_books")
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_books (
          book_id INTEGER PRIMARY KEY,
          completed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _embed_and_insert(
    conn: sqlite3.Connection,
    *,
    table: str,
    api_key: str,
    model: str,
    task: str,
    rows: list[ParagraphRow],
) -> None:
    res = _embed_payload(api_key=api_key, model=model, task=task, rows=rows)
    _insert_payload(conn, table=table, payload=res.payload)


def _embed_payload(*, api_key: str, model: str, task: str, rows: list[ParagraphRow]) -> EmbedResult:
    texts = [r.paragraph_text for r in rows]
    t0 = time.monotonic()
    vecs, total_tokens = _retry_embed_batch(api_key=api_key, model=model, task=task, texts=texts)
    embed_s = time.monotonic() - t0

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
    return EmbedResult(payload=payload, total_tokens=int(total_tokens), embed_s=float(embed_s))


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
    parser = argparse.ArgumentParser(description="Build embedding.db from Gutenberg text/ files (Jina embeddings API).")
    parser.add_argument("--text-dir", default="gutenberg/data/text", help="Directory of PG*_text.txt files.")
    parser.add_argument("--metadata-csv", default="gutenberg/metadata/metadata.csv", help="Metadata CSV with titles.")
    parser.add_argument("--out-db", default="embedding.db", help="Output SQLite path.")
    parser.add_argument("--table", default=os.environ.get("EMBEDDINGS_TABLE", "embeddings"))
    parser.add_argument("--model", default=os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v3"))
    parser.add_argument("--task", default=os.environ.get("JINA_EMBED_TASK", "text-matching"))
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
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=0,
        help="0 means no periodic checkpoint; otherwise checkpoint WAL every N completed books.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume by skipping books already recorded in embedded_books (default: true).",
    )
    parser.add_argument("--timing", action="store_true", help="Print timing breakdown at end.")
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-paragraphs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--overwrite", action="store_true", help="Drop and recreate the embeddings table.")
    args = parser.parse_args()

    resume = True if args.resume is None else bool(args.resume)

    if args.overwrite and (args.resume is True):
        print("Warning: --overwrite implies a fresh build; --resume will be ignored.", flush=True)

    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise SystemExit("Missing JINA_API_KEY.")

    root = Path(__file__).resolve().parents[1]
    text_dir = (root / args.text_dir).resolve()
    metadata_csv = (root / args.metadata_csv).resolve()
    out_db = (root / args.out_db).resolve()

    if not text_dir.exists():
        raise SystemExit(f"Missing text dir: {text_dir}")

    titles, downloads_by_id = _load_metadata(metadata_csv)

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
        checkpoint_frequency = max(0, int(args.checkpoint_frequency))
        task = str(args.task or "text-matching")

        max_books = max(0, int(args.max_books))
        text_files = list(_iter_text_files(text_dir))
        if downloads_by_id:
            by_popularity = sorted(
                ((bid, downloads_by_id.get(bid, -1)) for bid, _ in text_files),
                key=lambda item: item[1],
                reverse=True,
            )
            top_popular_ids = {bid for bid, _ in by_popularity[:2000]}
            text_files.sort(key=lambda item: (item[0] not in top_popular_ids, -downloads_by_id.get(item[0], -1)))
        if max_books:
            text_files = text_files[:max_books]
        total_books = len(text_files)
        if total_books == 0:
            raise SystemExit(f"No PG*_text.txt files found in {text_dir}.")

        inserted = 0
        seen = 0
        next_embed_id = 1

        start_time = time.monotonic()
        last_progress_print_time = 0.0
        last_rate_time = start_time
        last_rate_books_done = 0
        rate_s_per_book: float | None = None
        total_tokens = 0
        total_embed_s = 0.0
        total_insert_s = 0.0
        total_batches = 0

        if not args.overwrite and resume:
            row = conn.execute(f"SELECT COALESCE(MAX(embed_id), 0) FROM {args.table}").fetchone()
            if row is not None:
                next_embed_id = max(1, int(row[0]) + 1)

            embedded_books_count = conn.execute("SELECT COUNT(*) FROM embedded_books").fetchone()
            if embedded_books_count is not None and int(embedded_books_count[0]) == 0:
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO embedded_books (book_id, completed_at)
                    SELECT DISTINCT book_id, datetime('now')
                    FROM {args.table}
                    WHERE book_id IS NOT NULL
                    """
                )
                conn.commit()

            embedded_books_count = conn.execute("SELECT COUNT(*) FROM embedded_books").fetchone()
            already_done = int(embedded_books_count[0]) if embedded_books_count is not None else 0
            print(
                f"Resuming: {already_done}/{total_books} books already completed; next embed_id={next_embed_id}",
            )

        # Embed + insert: optionally in parallel (embedding requests only; SQLite writes stay single-threaded).
        def _format_rate() -> str:
            nonlocal last_rate_time, last_rate_books_done, rate_s_per_book
            now = time.monotonic()
            # Update rolling estimate every ~5 books (or when called after a skip/finish).
            if books_done - last_rate_books_done >= 5:
                dt = now - last_rate_time
                db = books_done - last_rate_books_done
                if dt > 0 and db > 0:
                    inst = dt / db
                    if rate_s_per_book is None:
                        rate_s_per_book = inst
                    else:
                        # Exponential moving average for stability.
                        alpha = 0.2
                        rate_s_per_book = alpha * inst + (1 - alpha) * rate_s_per_book
                last_rate_time = now
                last_rate_books_done = books_done

            if rate_s_per_book is None or books_done == 0:
                return ""
            return f" ({rate_s_per_book:.1f}s/book)"

        def _print_progress(*, force: bool = False) -> None:
            nonlocal last_progress_print_time
            now = time.monotonic()
            if not force and (now - last_progress_print_time) < 0.5:
                return
            last_progress_print_time = now

            elapsed_s = now - start_time
            skipped_part = f" (skipped {books_skipped})" if books_skipped else ""
            rate_part = _format_rate()
            print(
                f"{elapsed_s:7.1f}s | Books {books_done}/{total_books}{skipped_part}{rate_part} | Embedded {inserted} chunks | Tokens {total_tokens}",
                end="\r",
            )

        def insert_one_payload(res: EmbedResult) -> None:
            nonlocal inserted, total_tokens, total_embed_s, total_insert_s, total_batches
            t0 = time.monotonic()
            _insert_payload(conn, table=args.table, payload=res.payload)
            conn.commit()
            total_insert_s += time.monotonic() - t0
            inserted += len(res.payload)
            total_tokens += int(res.total_tokens)
            total_embed_s += float(res.embed_s)
            total_batches += 1
            _print_progress(force=False)

        inflight: set[concurrent.futures.Future[EmbedResult]] = set()
        books_done = 0
        books_skipped = 0

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
                insert_one_payload(_embed_payload(api_key=api_key, model=args.model, task=task, rows=rows))
                return
            inflight.add(pool.submit(_embed_payload, api_key=api_key, model=args.model, task=task, rows=rows))
            if len(inflight) >= max_inflight:
                drain_one(pool=pool)

        pool: concurrent.futures.Executor | None
        if concurrency <= 1:
            pool = None
        else:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)

        try:
            for book_id, path in text_files:
                if resume and not args.overwrite:
                    completed = conn.execute(
                        "SELECT 1 FROM embedded_books WHERE book_id = ? LIMIT 1",
                        (int(book_id),),
                    ).fetchone()
                    if completed is not None:
                        books_done += 1
                        books_skipped += 1
                        if checkpoint_frequency and (books_done % checkpoint_frequency == 0):
                            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        _print_progress(force=True)
                        continue

                title = titles.get(book_id)

                if resume and not args.overwrite:
                    # If we previously crashed mid-book, remove any partial rows for this book
                    # (we only consider a book complete if it's in embedded_books).
                    conn.execute(f"DELETE FROM {args.table} WHERE book_id = ?", (int(book_id),))
                    conn.commit()

                batch: list[ParagraphRow] = []
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
                if batch:
                    submit_batch(pool=pool, rows=batch)
                    batch = []

                if pool is not None and inflight:
                    drain_all(pool=pool)

                books_done += 1
                conn.execute(
                    "INSERT OR REPLACE INTO embedded_books (book_id, completed_at) VALUES (?, datetime('now'))",
                    (int(book_id),),
                )
                conn.commit()

                if checkpoint_frequency and (books_done % checkpoint_frequency == 0):
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

                _print_progress(force=True)

                if max_paragraphs and seen >= max_paragraphs:
                    break
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

        total_rows = conn.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()
        total_rows_int = int(total_rows[0]) if total_rows is not None else 0
        if total_rows_int == 0:
            raise SystemExit(f"No chunks found in {text_dir}.")

    print()
    print(
        f"OK: added {inserted} chunks; db has {total_rows_int} rows at {out_db} (table: {args.table}, model: {args.model}, task: {args.task})"
    )
    if args.timing:
        wall_s = time.monotonic() - start_time
        embed_avg_ms = (1000.0 * total_embed_s / total_batches) if total_batches else 0.0
        insert_avg_ms = (1000.0 * total_insert_s / total_batches) if total_batches else 0.0
        tok_per_s = (float(total_tokens) / wall_s) if wall_s > 0 else 0.0
        print(
            "Timing: "
            f"wall={wall_s:.1f}s, "
            f"batches={total_batches}, "
            f"embed_sum={total_embed_s:.1f}s (avg {embed_avg_ms:.0f}ms/batch), "
            f"insert_sum={total_insert_s:.1f}s (avg {insert_avg_ms:.0f}ms/batch), "
            f"tokens={total_tokens} ({tok_per_s:.1f} tok/s)"
        )


if __name__ == "__main__":
    main()
