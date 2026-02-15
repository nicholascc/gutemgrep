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

function isAsciiAlphaNum(ch) {
  if (!ch) return false;
  const code = ch.charCodeAt(0);
  return (
    (code >= 48 && code <= 57) || // 0-9
    (code >= 65 && code <= 90) || // A-Z
    (code >= 97 && code <= 122) // a-z
  );
}

function renderInlineMarkdown(text) {
  if (text == null) return null;
  const s = String(text);
  const out = [];
  let i = 0;

  function pushText(t) {
    if (!t) return;
    out.push(t);
  }

  // Parse `_italics_` into <em>, avoiding intra-word underscores like `foo_bar`.
  let last = 0;
  i = 0;
  while (i < s.length) {
    if (s[i] !== "_") {
      i += 1;
      continue;
    }
    const prev = i > 0 ? s[i - 1] : "";
    if (isAsciiAlphaNum(prev)) {
      i += 1;
      continue;
    }

    let j = i + 1;
    while (j < s.length) {
      if (s[j] !== "_") {
        j += 1;
        continue;
      }
      const next = j + 1 < s.length ? s[j + 1] : "";
      if (isAsciiAlphaNum(next)) {
        j += 1;
        continue;
      }
      break;
    }

    if (j >= s.length || s[j] !== "_" || j <= i + 1) {
      i += 1;
      continue;
    }

    const content = s.slice(i + 1, j);
    if (content.trim().length === 0) {
      i += 1;
      continue;
    }

    pushText(s.slice(last, i));
    out.push(<em key={`em-${i}`}>{content}</em>);
    last = j + 1;
    i = j + 1;
  }
  pushText(s.slice(last));

  return out.length ? out : s;
}

function useBodyNightMode(enabled) {
  useEffect(() => {
    const cls = "night-mode";
    if (enabled) document.body.classList.add(cls);
    else document.body.classList.remove(cls);
    return () => {
      document.body.classList.remove(cls);
    };
  }, [enabled]);
}

function Stars({ enabled }) {
  const [stars, setStars] = useState([]);

  useEffect(() => {
    if (!enabled) return;
    if (stars.length) return;
    const starCount = 100;
    const next = [];
    for (let i = 0; i < starCount; i++) {
      const size = Math.random() * 2 + 1;
      next.push({
        key: i,
        left: `${Math.random() * 100}%`,
        top: `${Math.random() * 100}%`,
        delay: `${Math.random() * 3}s`,
        size: `${size}px`,
      });
    }
    setStars(next);
  }, [enabled, stars.length]);

  return (
    <div className="stars" aria-hidden="true">
      {stars.map((s) => (
        <div
          key={s.key}
          className="star"
          style={{
            left: s.left,
            top: s.top,
            width: s.size,
            height: s.size,
            animationDelay: s.delay,
          }}
        />
      ))}
    </div>
  );
}

function Shell({ nightMode, children }) {
  useBodyNightMode(nightMode);
  return (
    <>
      <div className="background-illustration" aria-hidden="true" />
      <Stars enabled={nightMode} />
      <div className="app">{children}</div>
    </>
  );
}

function ResultCard({ result }) {
  return (
    <article className="resultCard">
      <div className="resultTitle">
        {result.book_title ? renderInlineMarkdown(result.book_title) : "Untitled"}
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

      {result.prev_text ? (
        <p className="para paraMuted">{renderInlineMarkdown(result.prev_text)}</p>
      ) : null}
      <p className="para paraMain">{renderInlineMarkdown(result.paragraph_text)}</p>
      {result.next_text ? (
        <p className="para paraMuted">{renderInlineMarkdown(result.next_text)}</p>
      ) : null}
    </article>
  );
}

function ResultsView({ uuid, results }) {
  return (
    <section className="resultsShell">
      <div className="resultsHeader">
        <div className="resultsTitle">Results</div>
        <div className="muted mono">
          <a href={`/${uuid}`}>/{uuid}</a>
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
  const [nightMode, setNightMode] = useState(false);
  const [extraVisible, setExtraVisible] = useState(false);

  useEffect(() => {
    const el = document.querySelector(".additional-content");
    if (!el) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setExtraVisible(true);
        });
      },
      { threshold: 0.1 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  async function runSearch(e) {
    e?.preventDefault?.();
    setError("");
    setLoading(true);
    try {
      const safeTopK = Number.isFinite(topK) ? topK : 10;
      const resp = await postQuery({ query, topK: safeTopK, includeContext: true });
      const uuid = resp?.uuid ? String(resp.uuid) : "";
      if (!uuid) throw new Error("Missing uuid in response");

      // Prefer loading results from the persisted query (queries.db) to match the share URL flow.
      window.location.assign(`/${encodeURIComponent(uuid)}`);
    } catch (err) {
      setError(err?.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  const examples = [
    "descriptions of storms and tempests",
    "passages of extreme philosophical pretension",
    "moments of sudden revelation",
    "declarations of love in Victorian novels",
    "villainous monologues",
    "vague, dreamlike descriptions",
  ];

  return (
    <Shell nightMode={nightMode}>
      <div className="hero-container">
        <h1 className="logo">
          <span className="initial">T</span>he Index
        </h1>

        <div className="search-container">
          <form className="search-form" onSubmit={runSearch}>
            <textarea
              id="query"
              className="search-input"
              value={query}
              placeholder="passages concerning the sea by moonlight..."
              rows={3}
              required
              onChange={(e) => {
                const next = e.target.value;
                setQuery(next);
                if (!nightMode && next.trim().length > 0) setNightMode(true);
              }}
            />

            <div className="advanced">
              <label htmlFor="topk">Top K</label>
              <input
                id="topk"
                className="mono"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>

            <button
              className={`search-button ${loading ? "loading" : ""}`}
              type="submit"
              disabled={loading || !query.trim()}
            >
              {loading ? "Searching…" : "Search"}
            </button>
          </form>
          {error ? <div className="error">{error}</div> : null}
        </div>
      </div>

      <div className={`additional-content ${extraVisible ? "visible" : ""}`}>
        <div className="content-section">
          <h2 className="section-title">About The Index</h2>
          <div className="section-content">
            <p>
              A semantic search engine over Project Gutenberg&apos;s public domain library. Every chunk has been embedded
              as a vector, creating a searchable mathematical space of literary language.
            </p>
            <p>
              Search by meaning, not keywords. Ask for &quot;passages about the ocean at night&quot; and discover
              descriptions that match the mood and content, regardless of exact phrasing.
            </p>
          </div>
        </div>

        <div className="content-section">
          <h2 className="section-title">Try These</h2>
          <div className="examples-grid">
            {examples.map((ex) => (
              <div
                key={ex}
                className="example-card"
                onClick={() => {
                  setQuery(ex);
                  if (!nightMode) setNightMode(true);
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setQuery(ex);
                    if (!nightMode) setNightMode(true);
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }
                }}
              >
                <div className="example-text">{ex}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Shell>
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
    <Shell nightMode={false}>
      <div className="hero-container" style={{ minHeight: "auto", paddingBottom: "2rem" }}>
        <h1 className="logo" style={{ marginBottom: "1.25rem" }}>
          <span className="initial">T</span>he Index
        </h1>
        <div className="muted mono">
          Saved query: /{token} · <a href="/">new search</a> ·{" "}
          <a href={`/${token}`}>permalink</a>
        </div>
      </div>

      {loading ? (
        <div className="resultsShell">
          <div className="muted">Loading…</div>
        </div>
      ) : null}
      {error ? (
        <div className="resultsShell">
          <div className="error">{error}</div>
        </div>
      ) : null}
      {data ? <ResultsView uuid={data.uuid} results={data.results || []} /> : null}
    </Shell>
  );
}

export default function App() {
  const token = useInitialToken();
  if (token) return <SavedQueryPage token={token} />;
  return <SearchPage />;
}
