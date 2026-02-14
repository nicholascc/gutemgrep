import React, { useEffect, useMemo, useState } from "react";
import { getSavedQuery, postQuery } from "./api.js";

function useInitialToken() {
  return useMemo(() => {
    if (typeof window !== "undefined" && window.__GUTEMGREP_TOKEN__) {
      return String(window.__GUTEMGREP_TOKEN__);
    }
    const path = window.location.pathname || "/";
    if (path === "/" || path === "") return null;
    return path.replace(/^\/+/, "");
  }, []);
}

function formatScore(score) {
  if (score == null || Number.isNaN(Number(score))) return "";
  return Number(score).toFixed(3);
}

function ResultCard({ result }) {
  return (
    <article className="resultCard">
      <div className="resultMeta">
        <div className="resultTitle">
          {result.book_title ? result.book_title : "Untitled"}
        </div>
        <div className="resultSub">
          <span className="mono">embed_id</span> {result.embed_id}
          {" · "}
          <span className="mono">score</span> {formatScore(result.score)}
          {result.book_id != null ? (
            <>
              {" · "}
              <span className="mono">book_id</span> {result.book_id}
            </>
          ) : null}
        </div>
      </div>

      {result.prev_text ? (
        <p className="para paraMuted">{result.prev_text}</p>
      ) : null}
      <p className="para paraMain">{result.paragraph_text}</p>
      {result.next_text ? (
        <p className="para paraMuted">{result.next_text}</p>
      ) : null}
    </article>
  );
}

function ResultsView({ uuid, results }) {
  return (
    <section className="results">
      <div className="resultsHeader">
        <div>
          <div className="h2">Results</div>
          <div className="muted">
            Share:{" "}
            <a className="mono" href={`/${uuid}`}>
              /{uuid}
            </a>
          </div>
        </div>
      </div>

      <div className="resultsList">
        {results.length === 0 ? (
          <div className="muted">No results.</div>
        ) : (
          results.map((r) => <ResultCard key={r.embed_id} result={r} />)
        )}
      </div>
    </section>
  );
}

function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  async function runSearch(e) {
    e?.preventDefault?.();
    setError("");
    setLoading(true);
    setData(null);
    try {
      const safeTopK = Number.isFinite(topK) ? topK : 10;
      const resp = await postQuery({ query, topK: safeTopK, includeContext: true });
      setData(resp);
      if (resp?.uuid) {
        window.history.replaceState({}, "", "/");
      }
    } catch (err) {
      setError(err?.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div className="brand">gutemgrep</div>
        <div className="muted">Project Gutenberg paragraph search</div>
      </header>

      <main className="main">
        <form className="search" onSubmit={runSearch}>
          <label className="label" htmlFor="query">
            Query
          </label>
          <textarea
            id="query"
            className="input"
            rows={4}
            value={query}
            placeholder="love, pretension, most erotic paragraphs on arxiv…"
            onChange={(e) => setQuery(e.target.value)}
          />

          <div className="row">
            <label className="label" htmlFor="topk">
              Top K
            </label>
            <input
              id="topk"
              className="input inputSmall mono"
              type="number"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
            <button className="button" type="submit" disabled={loading}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>

          {error ? <div className="error">{error}</div> : null}
        </form>

        {data ? <ResultsView uuid={data.uuid} results={data.results || []} /> : null}
      </main>
    </div>
  );
}

function SavedQueryPage({ token }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const resp = await getSavedQuery({ token });
        if (!cancelled) setData(resp);
      } catch (err) {
        if (!cancelled) setError(err?.message || "Failed to load saved query");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <div className="page">
      <header className="header">
        <div className="brand">gutemgrep</div>
        <div className="muted">
          Saved query: <span className="mono">/{token}</span> ·{" "}
          <a href="/">new search</a>
        </div>
      </header>

      <main className="main">
        {loading ? <div className="muted">Loading…</div> : null}
        {error ? <div className="error">{error}</div> : null}
        {data ? <ResultsView uuid={data.uuid} results={data.results || []} /> : null}
      </main>
    </div>
  );
}

export default function App() {
  const token = useInitialToken();
  if (token) return <SavedQueryPage token={token} />;
  return <SearchPage />;
}
