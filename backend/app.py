import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory


@dataclass(frozen=True)
class AppConfig:
    embedding_db_path: Path
    queries_db_path: Path
    embeddings_table: str
    openai_api_key: Optional[str]
    openai_embedding_model: str
    default_top_k: int
    static_dir: Optional[Path]


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
    conn.commit()


def _openai_embed_text(*, api_key: str, model: str, text: str) -> np.ndarray:
    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": text},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI embeddings error ({resp.status_code}): {resp.text}")
    data = resp.json()
    embedding = data["data"][0]["embedding"]
    return np.asarray(embedding, dtype=np.float32)


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
    return conn.execute(
        f"""
        SELECT
          embed_id,
          paragraph_text,
          prev_embed_id,
          next_embed_id,
          embedding,
          book_title,
          book_id
        FROM {table}
        """
    )


def _search_embeddings(
    *,
    embedding_db_path: Path,
    embeddings_table: str,
    query_vec: np.ndarray,
    top_k: int,
    include_context: bool,
) -> List[Dict[str, Any]]:
    with _connect_sqlite(embedding_db_path) as conn:
        if not _has_table(conn, embeddings_table):
            raise RuntimeError(
                f"Embedding DB missing table '{embeddings_table}'. Found: {_table_names(conn)}"
            )

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in _embedding_rows(conn, table=embeddings_table):
            blob = row["embedding"]
            if blob is None:
                continue
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

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in scored[: max(1, top_k)]]

        if not include_context:
            for item in top:
                item.pop("prev_embed_id", None)
                item.pop("next_embed_id", None)
            return top

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

        return top


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

        # Rehydrate paragraph/context details from embedding DB, without calling OpenAI again.
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
    config = AppConfig(
        embedding_db_path=_env_path("EMBEDDING_DB_PATH", root / "embedding.db", root=root),
        queries_db_path=_env_path("QUERIES_DB_PATH", root / "queries.db", root=root),
        embeddings_table=os.environ.get("EMBEDDINGS_TABLE", "embeddings"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_embedding_model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        default_top_k=int(os.environ.get("DEFAULT_TOP_K", "10")),
        static_dir=_env_optional_path("STATIC_DIR", root=root),
    )

    # Disable Flask's built-in static route (/static/...) so our explicit
    # `/static/<path:filename>` handler can serve from `STATIC_DIR`.
    app = Flask(__name__, static_folder=None)
    app.config["APP_CONFIG"] = config

    @app.get("/health")
    def health() -> Response:
        cfg: AppConfig = app.config["APP_CONFIG"]
        return jsonify(
            {
                "ok": True,
                "embedding_db_path": str(cfg.embedding_db_path),
                "queries_db_path": str(cfg.queries_db_path),
                "embeddings_table": cfg.embeddings_table,
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

        if not cfg.openai_api_key:
            return jsonify({"ok": False, "error": "OPENAI_API_KEY not set"}), 500

        try:
            query_vec = _openai_embed_text(
                api_key=cfg.openai_api_key,
                model=cfg.openai_embedding_model,
                text=query_text,
            )
            results = _search_embeddings(
                embedding_db_path=cfg.embedding_db_path,
                embeddings_table=cfg.embeddings_table,
                query_vec=query_vec,
                top_k=top_k_int,
                include_context=include_context_bool,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        query_uuid = str(uuid.uuid4())
        response_body = {"uuid": query_uuid, "results": results}
        try:
            _save_query(
                queries_db_path=cfg.queries_db_path,
                query_uuid=query_uuid,
                request_body={"query": query_text, "top_k": top_k_int, "include_context": include_context_bool},
                response_body=response_body,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to persist query: {e}"}), 500

        return jsonify(response_body)

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

    return app


app = create_app()
