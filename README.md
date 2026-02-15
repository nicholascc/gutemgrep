# gutemgrep

(this is an older pun name based on "gutenberg" + "mgrep", but it's called Quire now.)

The two of us have been thinking a lot about the future of the humanities. Literature and philosophy offer us entry into the minds of other humans, deepen our thought, and tie us to our common heritage; we believe they're extremely important, and worth investing in. 

As technologists, the question naturally arises: what new affordances do all these huge advances in NLP offer us? For Treehacks this year, we wanted to explore this question. We created Quire, an interactive embedding-based exploration tool for interfacing with classic works in the humanities.

A quire, in bookbinding, is a set of leaves stitched together. Likewise, quire.ink stitches together the Project Gutenberg archive — more than 70k books dating back to before 1105 BC — into a creative, experimental search interface, designed for navigating and connecting Gutenberg's huge library of classic texts.

In this devpost there's a bunch of detail about how we built it, but we think the best thing to do to understand a new type of tool is to use it! Try it out at https://quire.ink

--- 

### How we built Quire

We downloaded (most of) the Project Gutenberg digital archive, and embedded it paragraph-by-paragraph using ElasticSearch's jina-embeddings-v3 into a custom database. We implemented an efficient custom cosine similarity search algorithm that reduced query time by 50x (10s -> 50ms), then built a ton of features in React for interacting with the archive through the embeddings, serving everything from a single Flask server.

### What Quire does

The fundamental goal of Quire is to let you move laterally through texts, "stitching" them together and finding passages with common semantic aspects. The primary feature is the search bar, which takes in any text query, embeds it, and then runs our faster algorithm to find the 10 most similar passages across the entire Gutenberg archive. You can then open any of these passages to see the whole book it's contained within, see other similar passages, and, most importantly, **save it to your collection**. 

The collection is a sidebar space that accumulates passages of text. These can be passages from books, but can also be custom text, or text that you altered using a builtin LLM rewriter (e.g. you can "rewrite this passage with a male main character" if you want to search for similar passages with characters of a different gender). Once you've accumulated text in your collection, you can average across a group of paassages to create a combined average embedding vector, which you can use to search — this allows you to search for passages that are similar to a whole group of passages, rather than just a single passage at a time.


### What opportunities does this open up?

Quire lets you ask all sorts of new questions that were much harder to ask before.  Rather than relying on secondary literature and human sorting, you can now search passage-by-passage through this huge corpus of major works, and move through the embedding space using custom alterations and search vectors.

Some examples: 

- You can find the passages in human literature that are maximally romantic, or maximally philosophical, or both of those at once, by collecting passages that have those characteristics, combining their embeddings, and searching.
- You can find new primary sources for specific questions, e.g. how people understood 'sovereignty' throughout time, just by searching in natural language (e.g. 'philosophical passages on the question of sovereignty'). 
- You can discover precursor works and possible influences in poetry by embedding a corpus of a poet's work.

But to be honest, we think you won't really get what Quire is until you've used it. Try it out at https://quire.ink/! (works great on both mobile + desktop)



# Gutenberg Embedding Search — Design Doc

A semantic search engine over Project Gutenberg's library. Users type a query (or paste a passage), and the app returns the most similar paragraphs from the corpus, ranked by cosine similarity.

---

## Quickstart (Local)

Prereqs: Python 3.12+, `uv`, and Node.js + `npm`.

From the repo root:

1. Install Python deps:
   - `uv sync`
2. (Optional) Create a tiny fake Gutenberg dataset:
   - `python gutenberg/make_fake_dataset.py`
3. (Optional) Build `embedding.db` from the fake dataset (calls OpenAI embeddings API):
   - `export OPENAI_API_KEY=...`
   - `python tools/build_embedding_db.py --overwrite`
2. Build the frontend into `static/`:
   - `cd frontend`
   - `npm install`
   - `npm run build`
   - `cd ..`
3. Configure env vars:
   - `export FLASK_APP=./backend/app.py`
   - `export STATIC_DIR=./static`
   - `export OPENAI_API_KEY=...` (required for `POST /query`, and for `tools/build_embedding_db.py`)
   - Optional:
     - `export EMBEDDING_DB_PATH=./embedding.db`
     - `export QUERIES_DB_PATH=./queries.db`
     - `export EMBEDDINGS_TABLE=embeddings`
4. Run the server:
   - `uv run flask run`

Then open:
- `http://127.0.0.1:5000/` for the search UI
- `http://127.0.0.1:5000/health` for a quick config check

Notes:
- `queries.db` is created automatically on first query.
- You need a populated `embedding.db` table (default `embeddings`) for search to return results.

## Scripts

- `scripts/add_notable_author.py`: add a `notable_author` column to Gutenberg `metadata.csv` using a list of notable authors (see `scripts/README.md` for usage).

---

## Building `embedding.db` (Dev)

The backend searches `embedding.db` (SQLite) for a table (default `embeddings`) with:

- `embed_id` (INTEGER PRIMARY KEY)
- `paragraph_text` (TEXT)
- `prev_embed_id` (INTEGER nullable)
- `next_embed_id` (INTEGER nullable)
- `embedding` (BLOB; `float32` bytes)
- `book_title` (TEXT nullable)
- `book_id` (INTEGER nullable)

This repo includes a minimal dev pipeline:

1. Generate fake Gutenberg “processed” text files:
   - `python gutenberg/make_fake_dataset.py`
2. Embed paragraphs and write `embedding.db`:
   - `export OPENAI_API_KEY=...`
   - `python tools/build_embedding_db.py --overwrite`

Defaults:
- Reads from `gutenberg/data/text/`
- Uses `gutenberg/metadata/metadata.csv` for `book_title` (if present)
- Writes `embedding.db` at repo root
- Uses `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`)

## Architecture Overview

A single Flask application running on a VPS serves both the static frontend and the backend API.

```
                         ┌─────────────────────────────────────┐
                         │              VPS                     │
                         │                                     │
  Browser ──────────────►│  Flask App                           │
                         │  ├─ Static frontend (HTML/CSS/JS)   │
                         │  ├─ POST /query ──► OpenAI API      │
                         │  │    embed query    (text-embed-3)  │
                         │  │    brute-force cosine sim         │
                         │  │    return top-k results           │
                         │  │                                   │
                         │  ├─ embedding.db (SQLite)            │
                         │  └─ queries.db   (SQLite)            │
                         └─────────────────────────────────────┘
```

---

## 1. Data Pipeline

**Source:** Project Gutenberg mirror/bulk download (plain text files).

**Steps:**

1. **Download & convert** — Pull plain-text books from a Gutenberg mirror. Strip headers/footers (Gutenberg license boilerplate).
2. **Paragraph split** — Split each book into paragraphs by paragraph breaks (`\n\n`). Target ~50 tokens per chunk as a starting point; this will be tuned experimentally.
3. **Embed & store** — Send each chunk to OpenAI's `text-embedding-3-small` model. Store the embedding vector alongside the paragraph text and its neighbors in SQLite.

---

## 2. Embedding Model

**Model:** OpenAI `text-embedding-3-small` (pretrained, used via API).

No finetuning — we embed both corpus paragraphs and user queries with the same model, then rank by cosine similarity. A query and a passage are both just vectors.

---

## 3. Databases

### Embedding DB (`embedding.db` — SQLite)

Stores the full corpus of embedded paragraphs.

| Column | Type | Description |
|--------|------|-------------|
| `embed_id` | INTEGER PRIMARY KEY | Unique ID per chunk |
| `paragraph_text` | TEXT | The raw paragraph text |
| `prev_embed_id` | INTEGER | Embed ID of the previous paragraph (for reading context) |
| `next_embed_id` | INTEGER | Embed ID of the next paragraph (for reading context) |
| `embedding` | BLOB | The embedding vector (serialized float array) |
| `book_title` | TEXT | Source book title |
| `book_id` | INTEGER | Gutenberg book ID |

**Search method:** Brute-force cosine similarity — load all vectors, compute similarity against the query vector, return top-k. Good enough for hackathon scale.

### Saved Queries DB (`queries.db` — SQLite)

Stores every query for shareable links.

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | TEXT PRIMARY KEY | UUID for the query (used in URLs) |
| `embed_id` | INTEGER | Top result's embed_id (or query's own embed_id) |
| `request_body` | TEXT | The full POST request body (query text, params) |
| `aesthetic_url` | TEXT | Optional human-readable slug (nullable) |
| `created_at` | DATETIME | Timestamp |

---

## 4. Query Flow

1. User types a query into the frontend.
2. Frontend sends `POST /query` with the query text.
3. Flask backend calls OpenAI to embed the query.
4. Backend computes cosine similarity against all stored embeddings (brute force).
5. Top-k results returned with paragraph text + prev/next paragraph context.
6. A UUID is generated and the query is saved to `queries.db`.
7. Results page is accessible at `site.com/<uuid>` (HTML), and the page loads saved results via JSON.

---

## 5. Frontend

Served by the same Flask app.

**Implementation (current):**
- React app (Vite build) in `frontend/`, built assets output to `static/`.
- Search page at `/` posts to `POST /query`.
- Results render **full paragraph text** with `prev_text`/`next_text` shown above/below at lower opacity.
- Saved query pages at `/<token>` are server-rendered HTML (Jinja template) that bootstraps the React app.

---

## 6. URL Scheme & Aesthetic URLs

Every query gets a UUID. Results are accessible at:

```
site.com/<uuid>
```

**Stretch goal:** A tiny LLM generates a human-readable slug for each query, producing URLs like:

```
site.com/the-sea-remembers-nothing
```

The slug maps to the UUID on the backend. The `aesthetic_url` column in `queries.db` stores the mapping.

---

## 7. API Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET /` | Serve frontend HTML (`index.html`) | |
| `GET /static/<path>` | Serve built JS/CSS/assets | |
| `POST /query` | Submit a search query; returns results + UUID | |
| `GET /<token>` | Render saved query page (HTML) by UUID or slug | |
| `GET /api/query/<token>` | Retrieve saved query results (JSON) by UUID or slug | |
| `GET /book/by-embed/<embed_id>` | Return all paragraphs for a book (JSON) | |

### Current Backend Details (Implemented)

Backend lives at `backend/app.py`.

#### Configuration (env vars)

- `OPENAI_API_KEY` (required for `POST /query`)
- `EMBEDDING_DB_PATH` (default `./embedding.db`)
- `QUERIES_DB_PATH` (default `./queries.db`)
- `EMBEDDINGS_TABLE` (default `embeddings`)
- `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`)
- `DEFAULT_TOP_K` (default `10`)
- `STATIC_DIR` (optional; if set, `GET /` serves `index.html` from here and `GET /static/<path>` serves other files). If `STATIC_DIR` is relative, it is resolved relative to the repo root.

#### Routes

- `GET /health`: returns basic config + `ok: true`.
- `GET /`: if `STATIC_DIR` is set, serves `index.html`; otherwise returns a small JSON message.
- `GET /static/<path:filename>`: serves static files from `STATIC_DIR` (only if set).
- `POST /query`: embeds the query via OpenAI, brute-force cosine similarity against all rows in `embedding.db`, returns top-k results, and persists the query + results into `queries.db`.
- `GET /<token>`: renders an HTML page (Jinja template) for a saved query (UUID or slug).
- `GET /api/query/<token>`: loads saved results as JSON (UUID or slug). Never calls OpenAI.
- `GET /book/by-embed/<embed_id>`: finds the `book_id` for that `embed_id`, then returns all paragraphs for the book (ordered by `embed_id`).

#### `POST /query` request/response

Request JSON:

- `query` (string, required). (`text` is also accepted as an alias.)
- `top_k` (int, optional; default `DEFAULT_TOP_K`, clamped to `[1, 100]`)
- `include_context` (bool, optional; default `true`)

Response JSON:

```json
{
  "uuid": "…",
  "results": [
    {
      "embed_id": 123,
      "score": 0.42,
      "paragraph_text": "…",
      "prev_text": "…",
      "next_text": "…",
      "book_title": "…",
      "book_id": 1342
    }
  ]
}
```

Notes:

- Similarity is cosine similarity on `float32` vectors loaded from the `embedding` BLOB.
- If `include_context` is `false`, the response omits `prev_text`/`next_text`.

#### Persistence (`queries.db`)

On each `POST /query`, `backend/app.py` ensures (creates if missing) these tables:

- `queries` with columns:
  - `uuid` (TEXT PRIMARY KEY)
  - `embed_id` (INTEGER; the top result embed id)
  - `request_body` (TEXT; JSON)
  - `response_body` (TEXT; JSON) (optional; used if the column exists)
  - `aesthetic_url` (TEXT nullable) (not generated right now; only used for lookup if present)
  - `created_at` (DATETIME; ISO8601 UTC)
- `query_results` with columns:
  - `uuid` (TEXT)
  - `rank` (INTEGER)
  - `embed_id` (INTEGER)
  - `score` (REAL)

`GET /api/query/<token>` never calls OpenAI: it returns saved results. If `queries.response_body` exists and is populated, it is returned directly; otherwise results are rehydrated from `query_results` + `embedding.db` (to include paragraph/context/book metadata).

#### Embedding DB assumptions (`embedding.db`)

The search code expects a table (default name `embeddings`) with these columns:

- `embed_id` (INTEGER PRIMARY KEY)
- `paragraph_text` (TEXT)
- `prev_embed_id` (INTEGER nullable)
- `next_embed_id` (INTEGER nullable)
- `embedding` (BLOB; `float32` bytes, dimension inferred from length)
- `book_title` (TEXT nullable)
- `book_id` (INTEGER nullable)

---

## 8. Tech Stack

| Component | Choice |
|-----------|--------|
| Backend | Flask (Python) |
| Embedding model | OpenAI `text-embedding-3-small` |
| Database | SQLite (both embedding store and query store) |
| Vector search | Brute-force cosine similarity (NumPy) |
| Frontend | React (Vite build) served by Flask |
| Hosting | VPS |
| Data source | Project Gutenberg mirror (bulk download) |

---

## 9. Open Questions / Future Work

- **Chunk size tuning:** Starting at ~50 tokens per paragraph; experiment with larger/smaller chunks and overlapping windows.
- **Scale:** How much of Gutenberg to embed for the demo? Start with a subset (e.g. top 100–1000 books) and scale up.
- **Aesthetic URL generation:** Which tiny LLM to use for slug generation? Could be a small local model or a cheap API call.
- **Caching:** Cache query embeddings to avoid redundant OpenAI calls for repeated queries.
- **Frontend polish:** Result highlighting, reading mode, book-level browsing.

---

## Implementation Notes (Current)

- React frontend lives in `frontend/` (Vite).
- `npm run build` outputs to `static/` with assets served by Flask at `/static/...` (set `STATIC_DIR=static` or `STATIC_DIR=Repos/gutemgrep/static`).
- `GET /<uuid>` renders a Jinja template (`backend/templates/uuid.html`) which loads the React bundle; the page fetches saved results from `GET /api/query/<token>`.
