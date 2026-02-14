# gutemgrep

well. i would like to download {all of gutenberg, some of annas archive, some of arxiv}, embed every paragraph, and run search on it for arbitrary objectives

mostly i want vector scores for everythinggg

one thing i want is like. paragraphs that maximize for certain attributes: love, pretension, vagueness, evilness, &c. just to play around with

nicholas — Yesterday at 9:37 PMThursday, February 12, 2026 at 9:37 PM
“most erotic paragraphs on arxiv”

for books you can do scoring on every part of the book and get summary statistics. what predicts shifts toward action in novels? what is the usual sentiment arc in each genre? how does style change throughout works?

one thing i want is like. paragraphs that maximize for certain attributes: love, pretension, vagueness, evilness, &c. just to play around with

mostly for humanities, thinking project gutenberg as first application and see if we cant scale

annas archive is open—but hard to download

@harrison
my friend (albert huang) did something similar to this a while ago and actually made + sold a company with it

Recommendation: use paragraph-level embeddings (with light overlap) rather than maxing the embedding context window, because this preserves granularity for “top paragraphs” queries and cheaper, more interpretable per-book stats; pick a strong general-purpose model like `text-embedding-3-large` (or `text-embedding-3-small` for cost) or open-source `bge-large-en-v1.5`/`e5-large-v2`, and optionally store a second “contextual” vector that includes neighboring paragraphs for recall on ambiguous passages while keeping the primary index paragraph-only.

# Gutenberg Embedding Search — Design Doc

A semantic search engine over Project Gutenberg's library. Users type a query (or paste a passage), and the app returns the most similar paragraphs from the corpus, ranked by cosine similarity.

---

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
7. Results page is accessible at `site.com/<uuid>`.

---

## 5. Frontend

Served as static files by the same Flask app.

**Core features:**
- Search bar for queries
- Results list with paragraph snippets and book metadata
- Expandable context (prev/next paragraphs) for each result
- Shareable result URLs

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
| `GET /` | Serve frontend | |
| `POST /query` | Submit a search query, returns results + UUID | |
| `GET /<uuid>` | Retrieve saved query results | |
| `GET /<aesthetic-slug>` | Resolve slug → UUID → results | |

---

## 8. Tech Stack

| Component | Choice |
|-----------|--------|
| Backend | Flask (Python) |
| Embedding model | OpenAI `text-embedding-3-small` |
| Database | SQLite (both embedding store and query store) |
| Vector search | Brute-force cosine similarity (NumPy) |
| Frontend | Static HTML/CSS/JS served by Flask |
| Hosting | VPS |
| Data source | Project Gutenberg mirror (bulk download) |

---

## 9. Open Questions / Future Work

- **Chunk size tuning:** Starting at ~50 tokens per paragraph; experiment with larger/smaller chunks and overlapping windows.
- **Scale:** How much of Gutenberg to embed for the demo? Start with a subset (e.g. top 100–1000 books) and scale up.
- **Aesthetic URL generation:** Which tiny LLM to use for slug generation? Could be a small local model or a cheap API call.
- **Caching:** Cache query embeddings to avoid redundant OpenAI calls for repeated queries.
- **Frontend polish:** Result highlighting, reading mode, book-level browsing.
