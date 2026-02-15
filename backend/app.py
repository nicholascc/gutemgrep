import json
import os
import resource
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hnswlib = None


@dataclass(frozen=True)
class AppConfig:
    embedding_db_path: Path
    queries_db_path: Path
    embeddings_table: str
    jina_api_key: Optional[str]
    jina_embedding_model: str
    jina_embedding_task: str
    default_top_k: int
    static_dir: Optional[Path]
    perf_log_enabled: bool
    perf_include_in_response: bool
    hnsw_enabled: bool
    hnsw_warm_build: bool
    hnsw_index_path: Path
    hnsw_m: int
    hnsw_ef_construction: int
    hnsw_ef_search: int
    hnsw_log_every: int


def _require_book_metadata(conn: sqlite3.Connection) -> None:
    if not _has_table(conn, "book_metadata"):
        raise RuntimeError(
            "Missing book_metadata table; run tools/add_embedding_db_metadata.py to populate it."
        )


def _book_filter_sql(alias: str = "bm") -> str:
    return (
        f"{alias}.language IS NOT NULL "
        f"AND (lower({alias}.language) = 'en' OR lower({alias}.language) LIKE \"%'en'%\") "
        f"AND ({alias}.book_title IS NULL OR lower({alias}.book_title) NOT LIKE '%dictionary%')"
    )


def _cpu_count() -> int:
    return int(os.cpu_count() or 1)


def _read_proc_ints(path: Path) -> Dict[str, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    out: Dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        value = rest.strip().split(" ", 1)[0]
        try:
            out[key] = int(value)
        except ValueError:
            continue
    return out


def _read_process_io() -> Dict[str, int]:
    return _read_proc_ints(Path("/proc/self/io"))


def _read_meminfo() -> Dict[str, int]:
    return _read_proc_ints(Path("/proc/meminfo"))


def _process_rss_kb() -> Optional[int]:
    status = _read_proc_ints(Path("/proc/self/status"))
    value = status.get("VmRSS")
    return int(value) if value is not None else None


def _sample_perf_snapshot() -> Dict[str, Any]:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    proc_io = _read_process_io()
    return {
        "cpu_user_s": float(ru.ru_utime),
        "cpu_system_s": float(ru.ru_stime),
        "page_faults_major": int(ru.ru_majflt),
        "page_faults_minor": int(ru.ru_minflt),
        "ctx_switches_voluntary": int(ru.ru_nvcsw),
        "ctx_switches_involuntary": int(ru.ru_nivcsw),
        "rss_kb": _process_rss_kb(),
        "proc_io_read_bytes": int(proc_io.get("read_bytes", 0)),
        "proc_io_write_bytes": int(proc_io.get("write_bytes", 0)),
        "proc_io_rchar_bytes": int(proc_io.get("rchar", 0)),
        "proc_io_wchar_bytes": int(proc_io.get("wchar", 0)),
    }


def _snapshot_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    def _delta_int(name: str) -> int:
        return int(after.get(name, 0)) - int(before.get(name, 0))

    cpu_user = float(after.get("cpu_user_s", 0.0)) - float(before.get("cpu_user_s", 0.0))
    cpu_system = float(after.get("cpu_system_s", 0.0)) - float(before.get("cpu_system_s", 0.0))
    rss_before = before.get("rss_kb")
    rss_after = after.get("rss_kb")
    return {
        "cpu_user_s": cpu_user,
        "cpu_system_s": cpu_system,
        "cpu_total_s": cpu_user + cpu_system,
        "page_faults_major": _delta_int("page_faults_major"),
        "page_faults_minor": _delta_int("page_faults_minor"),
        "ctx_switches_voluntary": _delta_int("ctx_switches_voluntary"),
        "ctx_switches_involuntary": _delta_int("ctx_switches_involuntary"),
        "rss_kb_delta": (int(rss_after) - int(rss_before))
        if rss_before is not None and rss_after is not None
        else None,
        "proc_io_read_bytes": _delta_int("proc_io_read_bytes"),
        "proc_io_write_bytes": _delta_int("proc_io_write_bytes"),
        "proc_io_rchar_bytes": _delta_int("proc_io_rchar_bytes"),
        "proc_io_wchar_bytes": _delta_int("proc_io_wchar_bytes"),
    }


def _timed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path, *, root: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_path(name: str, *, root: Path) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _file_signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _db_signature(*, db_path: Path, embeddings_table: str) -> Dict[str, Any]:
    with _connect_sqlite(db_path) as conn:
        if not _has_table(conn, embeddings_table):
            raise RuntimeError(
                f"Embedding DB missing table '{embeddings_table}'. Found: {_table_names(conn)}"
            )
        _require_book_metadata(conn)
        row = conn.execute(
            f"SELECT length(embedding) AS blob_len FROM {embeddings_table} "
            "WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row is None or row["blob_len"] is None:
            raise RuntimeError("No embeddings found to build HNSW index.")
        blob_len = int(row["blob_len"])
        if blob_len % 4 != 0:
            raise RuntimeError(f"Unexpected embedding blob length: {blob_len}")
        dim = blob_len // 4

        count_row = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {embeddings_table} AS e
            JOIN book_metadata AS bm ON e.book_id = bm.book_id
            WHERE e.embedding IS NOT NULL AND {_book_filter_sql('bm')}
            """
        ).fetchone()
        total = int(count_row["n"]) if count_row and count_row["n"] is not None else 0

        data_version = conn.execute("PRAGMA data_version").fetchone()[0]
        schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]

    mtime_ns, size = _file_signature(db_path)
    return {
        "mtime_ns": int(mtime_ns),
        "size": int(size),
        "data_version": int(data_version),
        "schema_version": int(schema_version),
        "page_count": int(page_count),
        "freelist_count": int(freelist_count),
        "user_version": int(user_version),
        "embeddings_table": embeddings_table,
        "row_count": int(total),
        "dim": int(dim),
    }


def _sig_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".sig.json")


def _table_names(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [str(r["name"]) for r in rows]


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return table in set(_table_names(conn))


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    if not _has_table(conn, table):
        return []
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r["name"]) for r in rows]


def _ensure_queries_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queries (
          uuid TEXT PRIMARY KEY,
          embed_id INTEGER,
          request_body TEXT,
          response_body TEXT,
          aesthetic_url TEXT,
          created_at DATETIME
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_results (
          uuid TEXT NOT NULL,
          rank INTEGER NOT NULL,
          embed_id INTEGER NOT NULL,
          score REAL NOT NULL,
          PRIMARY KEY (uuid, rank)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_query_results_uuid ON query_results(uuid)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS custom_vectors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          embedding BLOB NOT NULL,
          source_embed_ids TEXT,
          source_texts TEXT,
          created_at DATETIME
        )
        """
    )
    conn.commit()


def _jina_embed_text(*, api_key: str, model: str, task: str, text: str) -> np.ndarray:
    resp = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "task": task, "input": [text], "truncate": True},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Jina embeddings error ({resp.status_code}): {resp.text}")
    data = resp.json()
    items = data.get("data") or []
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Unexpected embeddings response shape: {data}")
    embedding = items[0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"Unexpected embeddings response shape: {data}")
    return np.asarray(embedding, dtype=np.float32)


def _anthropic_single_turn(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")
    data = resp.json()
    content = data.get("content")
    if not isinstance(content, list):
        raise RuntimeError(f"Unexpected Anthropic response shape: {data}")
    parts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts).strip()


def _cosine_similarity(query_vec: np.ndarray, candidate_vec: np.ndarray) -> float:
    qn = float(np.linalg.norm(query_vec))
    cn = float(np.linalg.norm(candidate_vec))
    if qn == 0.0 or cn == 0.0:
        return float("-inf")
    return float(np.dot(query_vec, candidate_vec) / (qn * cn))


def _fetch_neighbor_texts(
    conn: sqlite3.Connection, *, table: str, neighbor_ids: Sequence[int]
) -> Dict[int, str]:
    ids = [int(i) for i in neighbor_ids if i is not None and int(i) > 0]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT embed_id, paragraph_text FROM {table} WHERE embed_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {int(r["embed_id"]): str(r["paragraph_text"]) for r in rows}


def _embedding_rows(
    conn: sqlite3.Connection, *, table: str
) -> Iterable[sqlite3.Row]:
    _require_book_metadata(conn)
    return conn.execute(
        f"""
        SELECT
          e.embed_id,
          e.paragraph_text,
          e.prev_embed_id,
          e.next_embed_id,
          e.embedding,
          e.book_title,
          e.book_id
        FROM {table} AS e
        JOIN book_metadata AS bm ON e.book_id = bm.book_id
        WHERE e.embedding IS NOT NULL AND {_book_filter_sql('bm')}
        """
    )


def _build_hnsw_index(
    *,
    embedding_db_path: Path,
    embeddings_table: str,
    m: int,
    ef_construction: int,
    ef_search: int,
    log_every: int,
    index_path: Optional[Path],
) -> Tuple["hnswlib.Index", Tuple[int, int]]:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed; disable HNSW or add the dependency.")
    signature = _db_signature(db_path=embedding_db_path, embeddings_table=embeddings_table)
    dim = int(signature["dim"])
    total = int(signature["row_count"])
    if total <= 0:
        raise RuntimeError("No embeddings found to build HNSW index.")

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=total, ef_construction=ef_construction, M=m)

    batch_ids: List[int] = []
    batch_vecs: List[np.ndarray] = []
    batch_size = 10000
    processed = 0
    log_every = max(int(log_every), 0)

    with _connect_sqlite(embedding_db_path) as conn:
        _require_book_metadata(conn)
        total_row = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {embeddings_table} AS e
            JOIN book_metadata AS bm ON e.book_id = bm.book_id
            WHERE e.embedding IS NOT NULL AND {_book_filter_sql('bm')}
            """
        ).fetchone()
        total = int(total_row["n"]) if total_row and total_row["n"] is not None else 0
        if total <= 0:
            raise RuntimeError("No embeddings found to build HNSW index (after filtering).")

        for row in conn.execute(
            f"""
            SELECT e.embed_id, e.embedding
            FROM {embeddings_table} AS e
            JOIN book_metadata AS bm ON e.book_id = bm.book_id
            WHERE e.embedding IS NOT NULL AND {_book_filter_sql('bm')}
            """
        ):
            blob = row["embedding"]
            if blob is None:
                continue
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size != dim:
                continue
            batch_ids.append(int(row["embed_id"]))
            batch_vecs.append(vec)
            if len(batch_ids) >= batch_size:
                index.add_items(np.vstack(batch_vecs), np.asarray(batch_ids, dtype=np.int64))
                processed += len(batch_ids)
                if log_every and processed % log_every == 0:
                    print(f"[hnsw] indexed {processed}/{total} vectors")
                batch_ids = []
                batch_vecs = []

    if batch_ids:
        index.add_items(np.vstack(batch_vecs), np.asarray(batch_ids, dtype=np.int64))
        processed += len(batch_ids)
        if log_every:
            print(f"[hnsw] indexed {processed}/{total} vectors")

    index.set_ef(max(ef_search, 1))

    if index_path is not None:
        sig_path = _sig_path(index_path)
        index.save_index(str(index_path))
        sig_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    return index, _file_signature(embedding_db_path)


def _load_hnsw_index(
    *,
    index_path: Path,
    embedding_db_path: Path,
    embeddings_table: str,
    ef_search: int,
) -> Optional["hnswlib.Index"]:
    if hnswlib is None:
        return None
    sig_path = _sig_path(index_path)
    if not index_path.exists() or not sig_path.exists():
        return None
    try:
        on_disk_sig = json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    current_sig = _db_signature(db_path=embedding_db_path, embeddings_table=embeddings_table)
    if on_disk_sig != current_sig:
        return None
    index = hnswlib.Index(space="cosine", dim=int(current_sig["dim"]))
    index.load_index(str(index_path))
    index.set_ef(max(ef_search, 1))
    return index


def _search_embeddings_ann(
    *,
    embedding_db_path: Path,
    embeddings_table: str,
    query_vec: np.ndarray,
    top_k: int,
    include_context: bool,
    index: "hnswlib.Index",
) -> List[Dict[str, Any]]:
    if query_vec.ndim != 1:
        query_vec = query_vec.reshape(-1)
    k = max(1, top_k)
    labels, distances = index.knn_query(query_vec, k=k)
    ids = [int(i) for i in labels[0].tolist() if int(i) >= 0]
    if not ids:
        return []
    scores = [1.0 - float(d) for d in distances[0].tolist()]
    score_by_id = {eid: score for eid, score in zip(ids, scores)}

    with _connect_sqlite(embedding_db_path) as conn:
        if not _has_table(conn, embeddings_table):
            raise RuntimeError(
                f"Embedding DB missing table '{embeddings_table}'. Found: {_table_names(conn)}"
            )
        _require_book_metadata(conn)
        placeholders = ",".join(["?"] * len(ids))
        rows = conn.execute(
            f"""
            SELECT
              embed_id,
              paragraph_text,
              prev_embed_id,
              next_embed_id,
              book_title,
              book_id
            FROM {embeddings_table}
            WHERE embed_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        by_id: Dict[int, sqlite3.Row] = {int(r["embed_id"]): r for r in rows}

        ordered: List[Dict[str, Any]] = []
        for eid in ids:
            row = by_id.get(eid)
            if row is None:
                continue
            ordered.append(
                {
                    "embed_id": int(row["embed_id"]),
                    "score": float(score_by_id.get(eid, float("-inf"))),
                    "paragraph_text": str(row["paragraph_text"]),
                    "prev_embed_id": int(row["prev_embed_id"]) if row["prev_embed_id"] is not None else None,
                    "next_embed_id": int(row["next_embed_id"]) if row["next_embed_id"] is not None else None,
                    "book_title": str(row["book_title"]) if row["book_title"] is not None else None,
                    "book_id": int(row["book_id"]) if row["book_id"] is not None else None,
                }
            )

        if not include_context:
            for item in ordered:
                item.pop("prev_embed_id", None)
                item.pop("next_embed_id", None)
            return ordered

        neighbor_ids: List[int] = []
        for item in ordered:
            if item.get("prev_embed_id") is not None:
                neighbor_ids.append(int(item["prev_embed_id"]))
            if item.get("next_embed_id") is not None:
                neighbor_ids.append(int(item["next_embed_id"]))
        neighbor_text = _fetch_neighbor_texts(conn, table=embeddings_table, neighbor_ids=neighbor_ids)

        for item in ordered:
            prev_id = item.pop("prev_embed_id", None)
            next_id = item.pop("next_embed_id", None)
            item["prev_text"] = neighbor_text.get(prev_id) if prev_id is not None else None
            item["next_text"] = neighbor_text.get(next_id) if next_id is not None else None

        return ordered


def _search_embeddings(
    *,
    embedding_db_path: Path,
    embeddings_table: str,
    query_vec: np.ndarray,
    top_k: int,
    include_context: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total_start = time.perf_counter()
    with _connect_sqlite(embedding_db_path) as conn:
        validate_start = time.perf_counter()
        if not _has_table(conn, embeddings_table):
            raise RuntimeError(
                f"Embedding DB missing table '{embeddings_table}'. Found: {_table_names(conn)}"
            )
        _require_book_metadata(conn)
        validate_ms = _timed_ms(validate_start)

        scan_start = time.perf_counter()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        rows_scanned = 0
        embedding_blob_bytes = 0
        for row in _embedding_rows(conn, table=embeddings_table):
            rows_scanned += 1
            blob = row["embedding"]
            if blob is None:
                continue
            embedding_blob_bytes += len(blob)
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size == 0:
                continue
            score = _cosine_similarity(query_vec, vec)
            scored.append(
                (
                    score,
                    {
                        "embed_id": int(row["embed_id"]),
                        "score": score,
                        "paragraph_text": str(row["paragraph_text"]),
                        "prev_embed_id": int(row["prev_embed_id"]) if row["prev_embed_id"] is not None else None,
                        "next_embed_id": int(row["next_embed_id"]) if row["next_embed_id"] is not None else None,
                        "book_title": str(row["book_title"]) if row["book_title"] is not None else None,
                        "book_id": int(row["book_id"]) if row["book_id"] is not None else None,
                    },
                )
            )
        scan_ms = _timed_ms(scan_start)

        sort_start = time.perf_counter()
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[: max(1, top_k)]]
        sort_ms = _timed_ms(sort_start)

        if not include_context:
            for item in top:
                item.pop("prev_embed_id", None)
                item.pop("next_embed_id", None)
            return (
                top,
                {
                    "rows_scanned": rows_scanned,
                    "embedding_blob_bytes_scanned": embedding_blob_bytes,
                    "db_validate_ms": validate_ms,
                    "scan_and_score_ms": scan_ms,
                    "sort_ms": sort_ms,
                    "context_fetch_ms": 0.0,
                    "total_ms": _timed_ms(total_start),
                },
            )

        context_start = time.perf_counter()
        neighbor_ids: List[int] = []
        for item in top:
            if item.get("prev_embed_id") is not None:
                neighbor_ids.append(int(item["prev_embed_id"]))
            if item.get("next_embed_id") is not None:
                neighbor_ids.append(int(item["next_embed_id"]))
        neighbor_text = _fetch_neighbor_texts(conn, table=embeddings_table, neighbor_ids=neighbor_ids)

        for item in top:
            prev_id = item.pop("prev_embed_id", None)
            next_id = item.pop("next_embed_id", None)
            item["prev_text"] = neighbor_text.get(prev_id) if prev_id is not None else None
            item["next_text"] = neighbor_text.get(next_id) if next_id is not None else None

        return (
            top,
            {
                "rows_scanned": rows_scanned,
                "embedding_blob_bytes_scanned": embedding_blob_bytes,
                "db_validate_ms": validate_ms,
                "scan_and_score_ms": scan_ms,
                "sort_ms": sort_ms,
                "context_fetch_ms": _timed_ms(context_start),
                "total_ms": _timed_ms(total_start),
            },
        )


def _save_query(
    *,
    queries_db_path: Path,
    query_uuid: str,
    request_body: Dict[str, Any],
    response_body: Dict[str, Any],
) -> None:
    with _connect_sqlite(queries_db_path) as conn:
        _ensure_queries_schema(conn)

        cols = set(_table_columns(conn, "queries"))
        created_at = datetime.now(timezone.utc).isoformat()
        top_embed_id = None
        results = response_body.get("results") or []
        if results:
            top_embed_id = results[0].get("embed_id")

        # Always persist normalized results so we can serve GET /<uuid> without recomputing.
        conn.execute("DELETE FROM query_results WHERE uuid = ?", (query_uuid,))
        for rank, r in enumerate(results):
            conn.execute(
                "INSERT INTO query_results(uuid, rank, embed_id, score) VALUES (?, ?, ?, ?)",
                (query_uuid, rank, int(r["embed_id"]), float(r["score"])),
            )

        insert_fields: Dict[str, Any] = {}
        if "uuid" in cols:
            insert_fields["uuid"] = query_uuid
        if "embed_id" in cols:
            insert_fields["embed_id"] = int(top_embed_id) if top_embed_id is not None else None
        if "request_body" in cols:
            insert_fields["request_body"] = json.dumps(request_body)
        if "response_body" in cols:
            insert_fields["response_body"] = json.dumps(response_body)
        if "created_at" in cols:
            insert_fields["created_at"] = created_at

        if insert_fields:
            columns_sql = ", ".join(insert_fields.keys())
            placeholders_sql = ", ".join(["?"] * len(insert_fields))
            conn.execute(
                f"INSERT OR REPLACE INTO queries ({columns_sql}) VALUES ({placeholders_sql})",
                list(insert_fields.values()),
            )

        conn.commit()


def _load_saved_query(
    *,
    queries_db_path: Path,
    embedding_db_path: Path,
    embeddings_table: str,
    token: str,
) -> Optional[Dict[str, Any]]:
    with _connect_sqlite(queries_db_path) as qconn:
        if not _has_table(qconn, "queries"):
            return None
        cols = set(_table_columns(qconn, "queries"))

        row = qconn.execute("SELECT * FROM queries WHERE uuid = ?", (token,)).fetchone()
        if row is None and "aesthetic_url" in cols:
            row = qconn.execute("SELECT * FROM queries WHERE aesthetic_url = ?", (token,)).fetchone()
        if row is None:
            return None

        if "response_body" in cols and row["response_body"]:
            data = json.loads(str(row["response_body"]))
            if "uuid" not in data:
                data["uuid"] = str(row["uuid"])
            return data

        uuid_value = str(row["uuid"])
        results_rows = qconn.execute(
            "SELECT rank, embed_id, score FROM query_results WHERE uuid = ? ORDER BY rank ASC",
            (uuid_value,),
        ).fetchall()
        if not results_rows:
            return {"uuid": uuid_value, "results": []}

        # Rehydrate paragraph/context details from embedding DB, without calling the embed API again.
        embed_ids = [int(r["embed_id"]) for r in results_rows]
        with _connect_sqlite(embedding_db_path) as econn:
            if not _has_table(econn, embeddings_table):
                raise RuntimeError(
                    f"Embedding DB missing table '{embeddings_table}'. Found: {_table_names(econn)}"
                )
            placeholders = ",".join(["?"] * len(embed_ids))
            rows = econn.execute(
                f"""
                SELECT
                  embed_id,
                  paragraph_text,
                  prev_embed_id,
                  next_embed_id,
                  book_title,
                  book_id
                FROM {embeddings_table}
                WHERE embed_id IN ({placeholders})
                """,
                embed_ids,
            ).fetchall()
            by_id: Dict[int, sqlite3.Row] = {int(r["embed_id"]): r for r in rows}

            neighbor_ids: List[int] = []
            for r in rows:
                if r["prev_embed_id"] is not None:
                    neighbor_ids.append(int(r["prev_embed_id"]))
                if r["next_embed_id"] is not None:
                    neighbor_ids.append(int(r["next_embed_id"]))
            neighbor_text = _fetch_neighbor_texts(econn, table=embeddings_table, neighbor_ids=neighbor_ids)

        results: List[Dict[str, Any]] = []
        for rr in results_rows:
            embed_id = int(rr["embed_id"])
            erow = by_id.get(embed_id)
            if erow is None:
                continue
            prev_id = int(erow["prev_embed_id"]) if erow["prev_embed_id"] is not None else None
            next_id = int(erow["next_embed_id"]) if erow["next_embed_id"] is not None else None
            results.append(
                {
                    "embed_id": embed_id,
                    "score": float(rr["score"]),
                    "paragraph_text": str(erow["paragraph_text"]),
                    "prev_text": neighbor_text.get(prev_id) if prev_id is not None else None,
                    "next_text": neighbor_text.get(next_id) if next_id is not None else None,
                    "book_title": str(erow["book_title"]) if erow["book_title"] is not None else None,
                    "book_id": int(erow["book_id"]) if erow["book_id"] is not None else None,
                }
            )

        return {"uuid": uuid_value, "results": results}


def create_app() -> Flask:
    root = _repo_root()
    embedding_db_path = _env_path("EMBEDDING_DB_PATH", root / "embedding.db", root=root)
    default_hnsw_index_path = _env_path(
        "HNSW_INDEX_PATH",
        embedding_db_path.with_suffix(".hnsw"),
        root=root,
    )
    config = AppConfig(
        embedding_db_path=embedding_db_path,
        queries_db_path=_env_path("QUERIES_DB_PATH", root / "queries.db", root=root),
        embeddings_table=os.environ.get("EMBEDDINGS_TABLE", "embeddings"),
        jina_api_key=os.environ.get("JINA_API_KEY"),
        jina_embedding_model=os.environ.get("JINA_EMBED_MODEL", "jina-embeddings-v3"),
        jina_embedding_task=os.environ.get("JINA_EMBED_TASK", "text-matching"),
        default_top_k=int(os.environ.get("DEFAULT_TOP_K", "10")),
        static_dir=_env_optional_path("STATIC_DIR", root=root),
        perf_log_enabled=_env_bool("PERF_LOG", True),
        perf_include_in_response=_env_bool("PERF_INCLUDE_IN_RESPONSE", False),
        hnsw_enabled=_env_bool("HNSW_ENABLED", True),
        hnsw_warm_build=_env_bool("HNSW_WARM_BUILD", True),
        hnsw_index_path=default_hnsw_index_path,
        hnsw_m=int(os.environ.get("HNSW_M", "16")),
        hnsw_ef_construction=int(os.environ.get("HNSW_EF_CONSTRUCTION", "200")),
        hnsw_ef_search=int(os.environ.get("HNSW_EF_SEARCH", "64")),
        hnsw_log_every=int(os.environ.get("HNSW_LOG_EVERY", "50000")),
    )

    # Disable Flask's built-in static route (/static/...) so our explicit
    # `/static/<path:filename>` handler can serve from `STATIC_DIR`.
    app = Flask(__name__, static_folder=None)
    app.config["APP_CONFIG"] = config
    app.config["HNSW_STATE"] = {
        "lock": threading.Lock(),
        "index": None,
        "file_sig": None,
    }
    if config.hnsw_enabled and config.hnsw_warm_build:
        if hnswlib is None:
            print("[hnsw] hnswlib not installed; skipping warm build")
        else:
            state = app.config["HNSW_STATE"]
            with state["lock"]:
                try:
                    print("[hnsw] warm build start")
                    loaded = _load_hnsw_index(
                        index_path=config.hnsw_index_path,
                        embedding_db_path=config.embedding_db_path,
                        embeddings_table=config.embeddings_table,
                        ef_search=config.hnsw_ef_search,
                    )
                    if loaded is not None:
                        state["index"] = loaded
                        state["file_sig"] = _file_signature(config.embedding_db_path)
                        print("[hnsw] warm build loaded from disk")
                    else:
                        index, file_sig = _build_hnsw_index(
                            embedding_db_path=config.embedding_db_path,
                            embeddings_table=config.embeddings_table,
                            m=config.hnsw_m,
                            ef_construction=config.hnsw_ef_construction,
                            ef_search=config.hnsw_ef_search,
                            log_every=config.hnsw_log_every,
                            index_path=config.hnsw_index_path,
                        )
                        state["index"] = index
                        state["file_sig"] = file_sig
                        print("[hnsw] warm build complete")
                except Exception as exc:
                    print(f"[hnsw] warm build failed: {exc}")

    @app.get("/health")
    def health() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        meminfo = _read_meminfo()
        return jsonify(
            {
                "ok": True,
                "embedding_db_path": str(cfg.embedding_db_path),
                "queries_db_path": str(cfg.queries_db_path),
                "embeddings_table": cfg.embeddings_table,
                "embed_provider": "jina",
                "jina_embedding_model": cfg.jina_embedding_model,
                "jina_embedding_task": cfg.jina_embedding_task,
                "has_jina_api_key": bool(cfg.jina_api_key),
                "cpu_count": _cpu_count(),
                "perf_log_enabled": cfg.perf_log_enabled,
                "perf_include_in_response": cfg.perf_include_in_response,
                "mem_total_kb": meminfo.get("MemTotal"),
                "mem_available_kb": meminfo.get("MemAvailable"),
                "hnsw_enabled": cfg.hnsw_enabled,
                "hnsw_warm_build": cfg.hnsw_warm_build,
                "hnsw_index_path": str(cfg.hnsw_index_path),
            }
        )

    @app.get("/debug/system")
    def debug_system() -> Response:
        meminfo = _read_meminfo()
        loadavg: Optional[Tuple[float, float, float]]
        try:
            load = os.getloadavg()
            loadavg = (float(load[0]), float(load[1]), float(load[2]))
        except Exception:
            loadavg = None
        return jsonify(
            {
                "ok": True,
                "cpu_count": _cpu_count(),
                "loadavg_1m_5m_15m": loadavg,
                "mem_total_kb": meminfo.get("MemTotal"),
                "mem_available_kb": meminfo.get("MemAvailable"),
                "swap_total_kb": meminfo.get("SwapTotal"),
                "swap_free_kb": meminfo.get("SwapFree"),
                "process_snapshot": _sample_perf_snapshot(),
                "hnsw_enabled": cfg.hnsw_enabled,
                "hnsw_warm_build": cfg.hnsw_warm_build,
                "hnsw_index_path": str(cfg.hnsw_index_path),
            }
        )

    @app.get("/")
    def index() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        if cfg.static_dir is None:
            return jsonify(
                {
                    "ok": True,
                    "message": "No STATIC_DIR configured; backend is running.",
                }
            )
        index_path = cfg.static_dir / "index.html"
        if not index_path.exists():
            return jsonify({"ok": False, "error": f"Missing {index_path}"}), 404
        return send_from_directory(str(cfg.static_dir), "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str) -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        if cfg.static_dir is None:
            return jsonify({"ok": False, "error": "STATIC_DIR not configured"}), 404
        return send_from_directory(str(cfg.static_dir), filename)

    @app.post("/query")
    def query() -> Response:
        req_start = time.perf_counter()
        perf_before = _sample_perf_snapshot()
        cfg: AppConfig = app.config["APP_CONFIG"]
        body = request.get_json(silent=True) or {}
        query_text = body.get("query") or body.get("text") or ""
        if not isinstance(query_text, str) or not query_text.strip():
            return jsonify({"ok": False, "error": "Missing non-empty 'query'"}), 400

        top_k = body.get("top_k", cfg.default_top_k)
        try:
            top_k_int = int(top_k)
        except Exception:
            return jsonify({"ok": False, "error": "'top_k' must be an integer"}), 400
        top_k_int = max(1, min(100, top_k_int))

        include_context = body.get("include_context", True)
        include_context_bool = bool(include_context)

        if not cfg.jina_api_key:
            return jsonify({"ok": False, "error": "JINA_API_KEY not set"}), 500

        try:
            embed_start = time.perf_counter()
            query_vec = _jina_embed_text(
                api_key=cfg.jina_api_key,
                model=cfg.jina_embedding_model,
                task=cfg.jina_embedding_task,
                text=query_text,
            )
            embed_ms = _timed_ms(embed_start)

            search_start = time.perf_counter()
            search_metrics: Dict[str, Any]
            if cfg.hnsw_enabled and hnswlib is not None:
                state = app.config["HNSW_STATE"]

                index_prepare_start = time.perf_counter()
                file_sig = _file_signature(cfg.embedding_db_path)
                index = state["index"]
                reloaded = False
                loaded_from_disk = False
                rebuilt = False

                if index is None or state["file_sig"] != file_sig:
                    with state["lock"]:
                        if state["index"] is None or state["file_sig"] != file_sig:
                            loaded = _load_hnsw_index(
                                index_path=cfg.hnsw_index_path,
                                embedding_db_path=cfg.embedding_db_path,
                                embeddings_table=cfg.embeddings_table,
                                ef_search=cfg.hnsw_ef_search,
                            )
                            if loaded is not None:
                                index = loaded
                                file_sig = _file_signature(cfg.embedding_db_path)
                                loaded_from_disk = True
                            else:
                                index, file_sig = _build_hnsw_index(
                                    embedding_db_path=cfg.embedding_db_path,
                                    embeddings_table=cfg.embeddings_table,
                                    m=cfg.hnsw_m,
                                    ef_construction=cfg.hnsw_ef_construction,
                                    ef_search=cfg.hnsw_ef_search,
                                    log_every=cfg.hnsw_log_every,
                                    index_path=cfg.hnsw_index_path,
                                )
                                rebuilt = True
                            state["index"] = index
                            state["file_sig"] = file_sig
                            reloaded = True

                index_prepare_ms = _timed_ms(index_prepare_start)

                ann_start = time.perf_counter()
                results = _search_embeddings_ann(
                    embedding_db_path=cfg.embedding_db_path,
                    embeddings_table=cfg.embeddings_table,
                    query_vec=query_vec,
                    top_k=top_k_int,
                    include_context=include_context_bool,
                    index=state["index"],
                )
                ann_ms = _timed_ms(ann_start)

                search_ms = _timed_ms(search_start)
                search_metrics = {
                    "method": "hnsw",
                    "index_prepare_ms": index_prepare_ms,
                    "ann_query_ms": ann_ms,
                    "reloaded": reloaded,
                    "loaded_from_disk": loaded_from_disk,
                    "rebuilt": rebuilt,
                    "total_ms": search_ms,
                }
            else:
                results, search_metrics = _search_embeddings(
                    embedding_db_path=cfg.embedding_db_path,
                    embeddings_table=cfg.embeddings_table,
                    query_vec=query_vec,
                    top_k=top_k_int,
                    include_context=include_context_bool,
                )
                search_ms = _timed_ms(search_start)
                search_metrics = {"method": "brute_force", **search_metrics}
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        query_uuid = str(uuid.uuid4())
        response_body = {"uuid": query_uuid, "results": results}
        try:
            persist_start = time.perf_counter()
            _save_query(
                queries_db_path=cfg.queries_db_path,
                query_uuid=query_uuid,
                request_body={"query": query_text, "top_k": top_k_int, "include_context": include_context_bool},
                response_body=response_body,
            )
            persist_ms = _timed_ms(persist_start)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to persist query: {e}"}), 500

        perf_after = _sample_perf_snapshot()
        wall_s = time.perf_counter() - req_start
        perf_delta = _snapshot_delta(perf_before, perf_after)
        perf_payload = {
            "wall_ms": wall_s * 1000.0,
            "wall_s": wall_s,
            "cpu_utilization_pct_single_core_estimate": (
                (perf_delta["cpu_total_s"] / wall_s) * 100.0 if wall_s > 0 else None
            ),
            "cpu_count": _cpu_count(),
            "timings_ms": {
                "embed_request_ms": embed_ms,
                "search_total_ms": search_ms,
                "persist_ms": persist_ms,
                "end_to_end_ms": wall_s * 1000.0,
            },
            "search_breakdown_ms": search_metrics,
            "resource_delta": perf_delta,
            "resource_after": perf_after,
        }

        if cfg.perf_log_enabled:
            app.logger.info("query_perf %s", json.dumps(perf_payload, sort_keys=True))

        include_perf = cfg.perf_include_in_response or bool(body.get("debug_perf"))
        if include_perf:
            response_body["perf"] = perf_payload

        return jsonify(response_body)

    @app.post("/haiku")
    def haiku() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"ok": False, "error": "Missing non-empty 'prompt'"}), 400

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return jsonify({"ok": False, "error": "ANTHROPIC_API_KEY not set"}), 500

        max_tokens = body.get("max_tokens", 512)
        try:
            max_tokens_int = int(max_tokens)
        except Exception:
            return jsonify({"ok": False, "error": "'max_tokens' must be an integer"}), 400
        max_tokens_int = max(1, min(8192, max_tokens_int))

        try:
            text = _anthropic_single_turn(
                api_key=api_key,
                model="claude-haiku-4-5",
                prompt=prompt,
                max_tokens=max_tokens_int,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

        return jsonify({"text": text})

    @app.get("/book/by-embed/<int:embed_id>")
    def book_by_embed(embed_id: int) -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        with _connect_sqlite(cfg.embedding_db_path) as conn:
            if not _has_table(conn, cfg.embeddings_table):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": f"Embedding DB missing table '{cfg.embeddings_table}'",
                        }
                    ),
                    500,
                )
            row = conn.execute(
                f"SELECT book_id, book_title FROM {cfg.embeddings_table} WHERE embed_id = ?",
                (embed_id,),
            ).fetchone()
            if row is None:
                return jsonify({"ok": False, "error": "embed_id not found"}), 404
            book_id = int(row["book_id"]) if row["book_id"] is not None else None
            book_title = str(row["book_title"]) if row["book_title"] is not None else None
            if book_id is None:
                return jsonify({"ok": False, "error": "Missing book_id for embed_id"}), 500

            paragraphs = conn.execute(
                f"""
                SELECT embed_id, paragraph_text
                FROM {cfg.embeddings_table}
                WHERE book_id = ?
                ORDER BY embed_id ASC
                """,
                (book_id,),
            ).fetchall()
            return jsonify(
                {
                    "book_id": book_id,
                    "book_title": book_title,
                    "paragraphs": [
                        {"embed_id": int(p["embed_id"]), "paragraph_text": str(p["paragraph_text"])}
                        for p in paragraphs
                    ],
                }
            )

    @app.get("/<string:token>")
    def saved_query_page(token: str) -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        if cfg.static_dir is None:
            return jsonify({"ok": False, "error": "STATIC_DIR not configured"}), 500

        try:
            data = _load_saved_query(
                queries_db_path=cfg.queries_db_path,
                embedding_db_path=cfg.embedding_db_path,
                embeddings_table=cfg.embeddings_table,
                token=token,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        if data is None:
            return render_template("uuid.html", token=token), 404

        return render_template("uuid.html", token=str(data.get("uuid") or token))

    @app.get("/api/query/<string:token>")
    def saved_query_or_slug_json(token: str) -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        try:
            data = _load_saved_query(
                queries_db_path=cfg.queries_db_path,
                embedding_db_path=cfg.embedding_db_path,
                embeddings_table=cfg.embeddings_table,
                token=token,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        if data is None:
            return jsonify({"ok": False, "error": "not found"}), 404
        return jsonify(data)

    # ── Custom Vectors ──────────────────────────────────────────────────────

    @app.post("/vectors/create")
    def create_vector() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        body = request.get_json(silent=True) or {}
        name = body.get("name", "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Missing 'name'"}), 400

        embed_ids: List[int] = [int(i) for i in (body.get("embed_ids") or [])]
        custom_texts: List[str] = [str(t) for t in (body.get("custom_texts") or []) if str(t).strip()]

        if not embed_ids and not custom_texts:
            return jsonify({"ok": False, "error": "Need at least one embed_id or custom_text"}), 400

        vectors: List[np.ndarray] = []

        # Fetch pre-embedded vectors from the database
        if embed_ids:
            with _connect_sqlite(cfg.embedding_db_path) as conn:
                placeholders = ",".join(["?"] * len(embed_ids))
                rows = conn.execute(
                    f"SELECT embed_id, embedding FROM {cfg.embeddings_table} "
                    f"WHERE embed_id IN ({placeholders})",
                    embed_ids,
                ).fetchall()
                for row in rows:
                    blob = row["embedding"]
                    if blob is not None:
                        vec = np.frombuffer(blob, dtype=np.float32).copy()
                        if vec.size > 0:
                            vectors.append(vec)

        # Embed custom texts via Jina
        if custom_texts:
            if not cfg.jina_api_key:
                return jsonify({"ok": False, "error": "JINA_API_KEY not set"}), 500
            for text in custom_texts:
                try:
                    vec = _jina_embed_text(
                        api_key=cfg.jina_api_key,
                        model=cfg.jina_embedding_model,
                        task=cfg.jina_embedding_task,
                        text=text,
                    )
                    vectors.append(vec)
                except Exception as e:
                    return jsonify({"ok": False, "error": f"Embedding failed: {e}"}), 500

        if not vectors:
            return jsonify({"ok": False, "error": "No valid embeddings found"}), 400

        # Average all vectors
        avg_vec = np.mean(np.vstack(vectors), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg_vec))
        if norm > 0:
            avg_vec = avg_vec / norm

        # Save to custom_vectors table
        created_at = datetime.now(timezone.utc).isoformat()
        with _connect_sqlite(cfg.queries_db_path) as conn:
            _ensure_queries_schema(conn)
            cursor = conn.execute(
                "INSERT INTO custom_vectors (name, embedding, source_embed_ids, source_texts, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    name,
                    avg_vec.tobytes(),
                    json.dumps(embed_ids) if embed_ids else None,
                    json.dumps(custom_texts) if custom_texts else None,
                    created_at,
                ),
            )
            vector_id = cursor.lastrowid
            conn.commit()

        return jsonify({
            "ok": True,
            "id": vector_id,
            "name": name,
            "source_embed_ids": embed_ids,
            "source_texts": custom_texts,
            "created_at": created_at,
        })

    @app.get("/vectors/<int:vector_id>")
    def get_vector(vector_id: int) -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        with _connect_sqlite(cfg.queries_db_path) as conn:
            _ensure_queries_schema(conn)
            row = conn.execute(
                "SELECT id, name, source_embed_ids, source_texts, created_at FROM custom_vectors WHERE id = ?",
                (vector_id,),
            ).fetchone()
            if row is None:
                return jsonify({"ok": False, "error": "Vector not found"}), 404
            return jsonify({
                "id": int(row["id"]),
                "name": str(row["name"]),
                "source_embed_ids": json.loads(row["source_embed_ids"]) if row["source_embed_ids"] else [],
                "source_texts": json.loads(row["source_texts"]) if row["source_texts"] else [],
                "created_at": str(row["created_at"]) if row["created_at"] else None,
            })

    @app.post("/vectors/list")
    def list_vectors() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        body = request.get_json(silent=True) or {}
        ids: List[int] = [int(i) for i in (body.get("ids") or [])]
        if not ids:
            return jsonify({"vectors": []})

        with _connect_sqlite(cfg.queries_db_path) as conn:
            _ensure_queries_schema(conn)
            placeholders = ",".join(["?"] * len(ids))
            rows = conn.execute(
                f"SELECT id, name, source_embed_ids, source_texts, created_at "
                f"FROM custom_vectors WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            vectors = []
            for row in rows:
                vectors.append({
                    "id": int(row["id"]),
                    "name": str(row["name"]),
                    "source_embed_ids": json.loads(row["source_embed_ids"]) if row["source_embed_ids"] else [],
                    "source_texts": json.loads(row["source_texts"]) if row["source_texts"] else [],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                })
            return jsonify({"vectors": vectors})

    @app.post("/query/by-vector")
    def query_by_vector() -> Response:
        req_start = time.perf_counter()
        perf_before = _sample_perf_snapshot()
        cfg: AppConfig = app.config["APP_CONFIG"]
        body = request.get_json(silent=True) or {}
        vector_id = body.get("vector_id")
        if vector_id is None:
            return jsonify({"ok": False, "error": "Missing 'vector_id'"}), 400

        top_k = body.get("top_k", cfg.default_top_k)
        try:
            top_k_int = int(top_k)
        except Exception:
            return jsonify({"ok": False, "error": "'top_k' must be an integer"}), 400
        top_k_int = max(1, min(100, top_k_int))

        include_context = body.get("include_context", True)
        include_context_bool = bool(include_context)

        # Load the custom vector and its source embed_ids
        with _connect_sqlite(cfg.queries_db_path) as conn:
            _ensure_queries_schema(conn)
            row = conn.execute(
                "SELECT embedding, source_embed_ids FROM custom_vectors WHERE id = ?",
                (int(vector_id),),
            ).fetchone()
            if row is None:
                return jsonify({"ok": False, "error": "Vector not found"}), 404
            query_vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            exclude_ids: set = set()
            if row["source_embed_ids"]:
                exclude_ids = set(json.loads(row["source_embed_ids"]))

        # Over-fetch to compensate for excluded source passages
        fetch_k = top_k_int + len(exclude_ids)

        try:
            search_start = time.perf_counter()
            search_metrics: Dict[str, Any]
            if cfg.hnsw_enabled and hnswlib is not None:
                state = app.config["HNSW_STATE"]

                index_prepare_start = time.perf_counter()
                file_sig = _file_signature(cfg.embedding_db_path)
                index = state["index"]

                if index is None or state["file_sig"] != file_sig:
                    with state["lock"]:
                        if state["index"] is None or state["file_sig"] != file_sig:
                            loaded = _load_hnsw_index(
                                index_path=cfg.hnsw_index_path,
                                embedding_db_path=cfg.embedding_db_path,
                                embeddings_table=cfg.embeddings_table,
                                ef_search=cfg.hnsw_ef_search,
                            )
                            if loaded is not None:
                                state["index"] = loaded
                                state["file_sig"] = _file_signature(cfg.embedding_db_path)
                            else:
                                idx, fs = _build_hnsw_index(
                                    embedding_db_path=cfg.embedding_db_path,
                                    embeddings_table=cfg.embeddings_table,
                                    m=cfg.hnsw_m,
                                    ef_construction=cfg.hnsw_ef_construction,
                                    ef_search=cfg.hnsw_ef_search,
                                    log_every=cfg.hnsw_log_every,
                                    index_path=cfg.hnsw_index_path,
                                )
                                state["index"] = idx
                                state["file_sig"] = fs

                index_prepare_ms = _timed_ms(index_prepare_start)

                ann_start = time.perf_counter()
                results = _search_embeddings_ann(
                    embedding_db_path=cfg.embedding_db_path,
                    embeddings_table=cfg.embeddings_table,
                    query_vec=query_vec,
                    top_k=fetch_k,
                    include_context=include_context_bool,
                    index=state["index"],
                )
                ann_ms = _timed_ms(ann_start)

                search_ms = _timed_ms(search_start)
                search_metrics = {
                    "method": "hnsw",
                    "index_prepare_ms": index_prepare_ms,
                    "ann_query_ms": ann_ms,
                    "total_ms": search_ms,
                }
            else:
                results, search_metrics = _search_embeddings(
                    embedding_db_path=cfg.embedding_db_path,
                    embeddings_table=cfg.embeddings_table,
                    query_vec=query_vec,
                    top_k=fetch_k,
                    include_context=include_context_bool,
                )
                search_ms = _timed_ms(search_start)
                search_metrics = {"method": "brute_force", **search_metrics}
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        # Filter out source passages and trim to requested top_k
        if exclude_ids:
            results = [r for r in results if r["embed_id"] not in exclude_ids]
        results = results[:top_k_int]

        query_uuid = str(uuid.uuid4())
        response_body = {"uuid": query_uuid, "results": results, "vector_id": int(vector_id)}

        try:
            persist_start = time.perf_counter()
            _save_query(
                queries_db_path=cfg.queries_db_path,
                query_uuid=query_uuid,
                request_body={"vector_id": int(vector_id), "top_k": top_k_int, "include_context": include_context_bool},
                response_body=response_body,
            )
            persist_ms = _timed_ms(persist_start)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to persist query: {e}"}), 500

        perf_after = _sample_perf_snapshot()
        wall_s = time.perf_counter() - req_start
        perf_delta = _snapshot_delta(perf_before, perf_after)
        perf_payload = {
            "wall_ms": wall_s * 1000.0,
            "wall_s": wall_s,
            "timings_ms": {
                "search_total_ms": search_ms,
                "persist_ms": persist_ms,
                "end_to_end_ms": wall_s * 1000.0,
            },
            "search_breakdown_ms": search_metrics,
            "resource_delta": perf_delta,
        }

        if cfg.perf_log_enabled:
            app.logger.info("vector_query_perf %s", json.dumps(perf_payload, sort_keys=True))

        include_perf = cfg.perf_include_in_response or bool(body.get("debug_perf"))
        if include_perf:
            response_body["perf"] = perf_payload

        return jsonify(response_body)

    return app


app = create_app()
