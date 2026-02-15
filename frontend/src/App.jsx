import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { postQuery, getSavedQuery, getBookByEmbed } from "./api.js";

// ─── Constants ───────────────────────────────────────────────────────────────

const EPIGRAPHS = [
  { text: "The universe (which others call the Library) is composed of an indefinite and perhaps infinite number of hexagonal galleries.", author: "Borges" },
  { text: "The limits of my language mean the limits of my world.", author: "Wittgenstein" },
  { text: "Every word was once a poem.", author: "Emerson" },
];

const FONT = "'EB Garamond', Garamond, 'Times New Roman', serif";
const WARM = (a) => `rgba(200, 185, 160, ${a})`;
const PARCHMENT = (a) => `rgba(225, 215, 195, ${a})`;
const DIM = (a) => `rgba(210, 198, 175, ${a})`;

// ─── Token from URL ──────────────────────────────────────────────────────────

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

// ─── Star Field ──────────────────────────────────────────────────────────────

function StarField() {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let w = (canvas.width = window.innerWidth);
    let h = (canvas.height = window.innerHeight);

    const stars = [];
    for (let i = 0; i < 400; i++) {
      stars.push({
        x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.2 + 0.2, depth: Math.random(),
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        twinkleOffset: Math.random() * Math.PI * 2,
      });
    }

    let time = 0;
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      time += 1;
      for (const star of stars) {
        const twinkle = 0.4 + 0.6 * Math.sin(time * star.twinkleSpeed + star.twinkleOffset);
        const alpha = (0.15 + star.depth * 0.65) * twinkle;
        ctx.fillStyle = star.depth > 0.7
          ? `rgba(255, 240, 220, ${alpha})`
          : `rgba(200, 210, 240, ${alpha})`;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
        ctx.fill();
      }
      animRef.current = requestAnimationFrame(draw);
    };
    draw();

    const onResize = () => { w = canvas.width = window.innerWidth; h = canvas.height = window.innerHeight; };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(animRef.current); window.removeEventListener("resize", onResize); };
  }, []);

  return <canvas ref={canvasRef} style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none" }} />;
}

function Vignette() {
  return <div style={{ position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none", background: "radial-gradient(ellipse at center, transparent 40%, rgba(6,8,18,0.5) 75%, rgba(6,8,18,0.95) 100%)" }} />;
}

// ─── Corner Expand/Collapse Button ───────────────────────────────────────────

function CornerButton({ expanded, onClick }) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      aria-label={expanded ? "Collapse back to search" : "Read in book"}
      style={{
        position: "absolute", top: 0, right: 0,
        width: 48, height: 48,
        background: hovered ? "rgba(180, 165, 140, 0.14)" : "rgba(180, 165, 140, 0.04)",
        border: "none",
        borderLeft: `1px solid ${WARM(hovered ? 0.18 : 0.08)}`,
        borderBottom: `1px solid ${WARM(hovered ? 0.18 : 0.08)}`,
        borderRadius: "0 3px 0 6px",
        cursor: "pointer",
        display: "flex", alignItems: "center", justifyContent: "center",
        transition: "all 0.25s ease",
        zIndex: 5,
      }}
    >
      <svg
        width="20" height="20" viewBox="0 0 20 20" fill="none"
        style={{
          transition: "transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)",
          transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
        }}
      >
        <path
          d="M6 14L14 6"
          stroke={WARM(hovered ? 0.9 : 0.5)}
          strokeWidth="2" strokeLinecap="round"
        />
        <path
          d="M8 6L14 6L14 12"
          stroke={WARM(hovered ? 0.9 : 0.5)}
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}

// ─── Lateral Carousel (cosine-similar passages) ──────────────────────────────

function SimilarCarousel({ embedId, allResults, onNavigate }) {
  const items = useMemo(() => {
    if (!allResults || allResults.length === 0) return [];
    return allResults
      .filter(x => x.embed_id !== embedId)
      .map(x => ({
        embed_id: x.embed_id, book_id: x.book_id,
        score: x.score,
        paragraph_text: x.paragraph_text,
        book_title: x.book_title,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);
  }, [embedId, allResults]);

  if (items.length === 0) return null;

  return (
    <div>
      <div style={{
        fontFamily: FONT, fontStyle: "italic", fontSize: 13,
        color: WARM(0.35), marginBottom: 12, paddingLeft: 2,
      }}>
        {"similar passages \u2190 \u2192"}
      </div>
      <div style={{
        display: "flex", gap: 14, overflowX: "auto", paddingBottom: 12,
        scrollSnapType: "x mandatory",
      }}>
        {items.map((item) => (
          <SimilarCard key={item.embed_id} item={item} onClick={() => onNavigate(item)} />
        ))}
      </div>
    </div>
  );
}

function SimilarCard({ item, onClick }) {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        flex: "0 0 300px", scrollSnapAlign: "start",
        background: hovered ? "rgba(22, 20, 28, 0.85)" : "rgba(22, 20, 28, 0.55)",
        border: `1px solid ${WARM(hovered ? 0.2 : 0.08)}`,
        borderRadius: 3, padding: "16px 18px",
        cursor: "pointer",
        transition: "all 0.25s ease",
      }}
    >
      <p style={{
        fontFamily: FONT, fontSize: 14.5, lineHeight: 1.7,
        color: PARCHMENT(hovered ? 0.75 : 0.55), margin: "0 0 10px 0",
        transition: "color 0.25s ease",
      }}>
        {item.paragraph_text.length > 150 ? item.paragraph_text.slice(0, 150) + "\u2026" : item.paragraph_text}
      </p>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 12, color: WARM(0.45) }}>
          {item.book_title}
        </span>
        <span style={{ fontFamily: FONT, fontSize: 11, color: WARM(0.25), fontVariantNumeric: "tabular-nums" }}>
          {item.score.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

// ─── Book Reader (expanded mode) ─────────────────────────────────────────────

function BookReader({ result, allResults, onCollapse, onTextSelect, onNavigateToSimilar, onHighlightMeasured }) {
  const scrollRef = useRef(null);
  const highlightRef = useRef(null);
  const [paragraphs, setParagraphs] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadBook() {
      try {
        const data = await getBookByEmbed({ embedId: result.embed_id });
        if (cancelled) return;
        const paras = (data.paragraphs || []).map(p => ({
          embed_id: p.embed_id,
          text: p.paragraph_text,
        }));
        setParagraphs(paras.length > 0 ? paras : [
          { embed_id: result.embed_id, text: result.paragraph_text },
        ]);
        setLoaded(true);
      } catch {
        if (cancelled) return;
        setParagraphs([{ embed_id: result.embed_id, text: result.paragraph_text }]);
        setLoaded(true);
      }
    }
    loadBook();
    return () => { cancelled = true; };
  }, [result.embed_id, result.paragraph_text]);

  useEffect(() => {
    if (loaded && highlightRef.current && scrollRef.current) {
      const container = scrollRef.current;
      const el = highlightRef.current;
      const offset = el.offsetTop - container.offsetTop - container.clientHeight * 0.3;
      container.scrollTop = Math.max(0, offset);

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (highlightRef.current && onHighlightMeasured) {
            const r = highlightRef.current.getBoundingClientRect();
            onHighlightMeasured({ top: r.top, left: r.left, width: r.width, height: r.height });
          }
        });
      });
    }
  }, [loaded, onHighlightMeasured]);

  const handleMouseUp = () => {
    const selection = window.getSelection();
    const text = selection?.toString().trim();
    if (text && text.length > 5) {
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      onTextSelect(text, { x: rect.left + rect.width / 2, y: rect.top });
    }
  };

  return (
    <div style={{
      width: "100%", maxWidth: 720, margin: "0 auto",
      display: "flex", flexDirection: "column",
      height: "100vh", padding: "20px 24px 0 24px",
    }}>
      {/* Header bar */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: 0, flexShrink: 0, paddingBottom: 16,
        borderBottom: `1px solid ${WARM(0.06)}`,
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 20, color: WARM(0.8), letterSpacing: "0.02em" }}>
            {result.book_title}
          </span>
          {result.book_id && (
            <span style={{ fontFamily: FONT, fontSize: 12, color: WARM(0.3), marginLeft: 12 }}>
              #{result.book_id}
            </span>
          )}
        </div>

        <button
          onClick={onCollapse}
          style={{
            background: WARM(0.06), border: `1px solid ${WARM(0.12)}`,
            borderRadius: 3, width: 44, height: 44, flexShrink: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            cursor: "pointer", transition: "all 0.25s ease",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = WARM(0.14); e.currentTarget.style.borderColor = WARM(0.25); }}
          onMouseLeave={(e) => { e.currentTarget.style.background = WARM(0.06); e.currentTarget.style.borderColor = WARM(0.12); }}
          aria-label="Collapse back to search"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none"
            style={{ transform: "rotate(180deg)" }}
          >
            <path d="M6 14L14 6" stroke={WARM(0.65)} strokeWidth="2" strokeLinecap="round" />
            <path d="M8 6L14 6L14 12" stroke={WARM(0.65)} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {/* Scrollable book content */}
      <div
        ref={scrollRef}
        onMouseUp={handleMouseUp}
        style={{
          flex: 1, overflowY: "auto", padding: "24px 0 60px 0",
          maskImage: "linear-gradient(to bottom, transparent 0%, black 2.5%, black 90%, transparent 100%)",
          WebkitMaskImage: "linear-gradient(to bottom, transparent 0%, black 2.5%, black 90%, transparent 100%)",
        }}
      >
        {!loaded && (
          <p style={{
            fontFamily: FONT, fontStyle: "italic", fontSize: 15,
            color: WARM(0.3), textAlign: "center", marginTop: 80,
            animation: "pulseGlow 1.5s ease infinite",
          }}>
            {"loading book\u2026"}
          </p>
        )}
        {loaded && paragraphs.map((para) => {
          const isHighlight = para.embed_id === result.embed_id;
          return (
            <BookParagraphWithRef key={para.embed_id} para={para} isHighlight={isHighlight} innerRef={isHighlight ? highlightRef : null} />
          );
        })}
      </div>

      {/* Lateral carousel pinned at bottom */}
      <div style={{
        flexShrink: 0, borderTop: `1px solid ${WARM(0.06)}`,
        paddingTop: 14, paddingBottom: 16,
      }}>
        <SimilarCarousel
          embedId={result.embed_id}
          allResults={allResults}
          onNavigate={onNavigateToSimilar}
        />
      </div>
    </div>
  );
}

const BookParagraphWithRef = ({ para, isHighlight, innerRef }) => {
  const [hovered, setHovered] = useState(false);
  const baseColor = isHighlight ? PARCHMENT(0.92) : DIM(0.4);
  const hoverColor = isHighlight ? PARCHMENT(0.95) : DIM(0.72);

  return (
    <div
      ref={innerRef}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ padding: "10px 28px 10px 32px", position: "relative", cursor: "text" }}
    >
      {isHighlight && (
        <div style={{
          position: "absolute", left: 6, top: 4, bottom: 4, width: 3,
          background: `linear-gradient(to bottom, transparent, ${WARM(0.55)}, transparent)`,
          borderRadius: 2,
        }} />
      )}
      <p style={{
        fontFamily: FONT, fontSize: isHighlight ? 18.5 : 17,
        lineHeight: 1.82, color: hovered ? hoverColor : baseColor,
        margin: 0, letterSpacing: "0.01em", transition: "color 0.3s ease",
      }}>
        {para.text}
      </p>
    </div>
  );
};

// ─── Selection Tooltip ───────────────────────────────────────────────────────

function SelectionTooltip({ text, position, onSearch }) {
  if (!text || !position) return null;
  return (
    <div style={{ position: "fixed", left: position.x, top: position.y - 48, transform: "translateX(-50%)", zIndex: 100, animation: "fadeInUp 0.25s ease" }}>
      <button
        onClick={() => onSearch(text)}
        style={{
          background: "rgba(30, 27, 38, 0.95)", border: `1px solid ${WARM(0.25)}`,
          borderRadius: 3, padding: "7px 16px",
          fontFamily: FONT, fontStyle: "italic", fontSize: 13, color: PARCHMENT(0.85),
          cursor: "pointer", backdropFilter: "blur(8px)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.4)", whiteSpace: "nowrap",
          transition: "border-color 0.2s ease",
        }}
        onMouseEnter={(e) => (e.target.style.borderColor = WARM(0.5))}
        onMouseLeave={(e) => (e.target.style.borderColor = WARM(0.25))}
      >
        {"search \u201c"}{text.length > 28 ? text.slice(0, 28) + "\u2026" : text}{"\u201d"}
      </button>
    </div>
  );
}

// ─── Collection Sidebar ──────────────────────────────────────────────────────

function CollectionSidebar({ items, open, onClose, onSearchAverage }) {
  return (
    <div style={{
      position: "fixed", right: 0, top: 0, bottom: 0, width: 360,
      background: "rgba(12, 10, 20, 0.95)", backdropFilter: "blur(20px)",
      borderLeft: `1px solid ${WARM(0.08)}`, zIndex: 50,
      transform: open ? "translateX(0)" : "translateX(100%)",
      transition: "transform 0.4s ease", padding: "40px 28px", overflowY: "auto",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 32 }}>
        <h3 style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 18, color: WARM(0.8), margin: 0, fontWeight: 400 }}>
          collected passages
        </h3>
        <button onClick={onClose} style={{ background: "none", border: "none", fontFamily: FONT, fontSize: 14, color: WARM(0.4), cursor: "pointer" }}>
          close
        </button>
      </div>
      {items.length === 0 ? (
        <p style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.3), lineHeight: 1.7 }}>
          {"Click \u201c+ collect\u201d on any passage to build a set. Search the centroid of your collection \u2014 the average embedding of everything you\u2019ve gathered."}
        </p>
      ) : (
        <>
          {items.map((item, i) => (
            <div key={i} style={{ padding: "14px 0", borderBottom: `1px solid ${WARM(0.06)}` }}>
              <p style={{ fontFamily: FONT, fontSize: 14, lineHeight: 1.65, color: DIM(0.7), margin: "0 0 6px 0" }}>
                {item.paragraph_text.slice(0, 120)}{"\u2026"}
              </p>
              <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 12, color: WARM(0.35) }}>{item.book_title}</span>
            </div>
          ))}
          <button
            onClick={onSearchAverage}
            style={{
              marginTop: 24, width: "100%", background: WARM(0.08),
              border: `1px solid ${WARM(0.15)}`, borderRadius: 3,
              padding: "10px 16px", fontFamily: FONT, fontStyle: "italic",
              fontSize: 14, color: WARM(0.7), cursor: "pointer", transition: "all 0.3s ease",
            }}
            onMouseEnter={(e) => { e.target.style.background = WARM(0.12); e.target.style.borderColor = WARM(0.25); }}
            onMouseLeave={(e) => { e.target.style.background = WARM(0.08); e.target.style.borderColor = WARM(0.15); }}
          >
            search centroid of collection
          </button>
        </>
      )}
    </div>
  );
}

// ─── Result Card ─────────────────────────────────────────────────────────────

function ResultCard({ result, index, onTextSelect, onExpand, onCollect }) {
  const [visible, setVisible] = useState(false);
  const cardBodyRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 80 * index + 100);
    return () => clearTimeout(timer);
  }, [index]);

  const handleMouseUp = () => {
    const selection = window.getSelection();
    const text = selection?.toString().trim();
    if (text && text.length > 5) {
      const range = selection.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      onTextSelect(text, { x: rect.left + rect.width / 2, y: rect.top });
    }
  };

  const handleExpand = () => {
    const el = cardBodyRef.current;
    if (el) {
      const r = el.getBoundingClientRect();
      onExpand(result, { top: r.top, left: r.left, width: r.width, height: r.height });
    } else {
      onExpand(result, null);
    }
  };

  return (
    <div
      onMouseUp={handleMouseUp}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(24px)",
        transition: "opacity 0.7s ease, transform 0.7s ease",
        marginBottom: 36, position: "relative",
      }}
    >
      {/* Book title + score */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10, padding: "0 2px" }}>
        <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.7), letterSpacing: "0.02em" }}>
          {result.book_title}
          {result.book_id != null && <span style={{ color: WARM(0.35), marginLeft: 8, fontSize: 12, fontStyle: "normal" }}>#{result.book_id}</span>}
        </span>
        <span style={{ fontFamily: FONT, fontSize: 12, color: WARM(0.35), fontVariantNumeric: "tabular-nums" }}>
          {result.score.toFixed(3)}
        </span>
      </div>

      {/* Card body */}
      <div ref={cardBodyRef} style={{
        background: "rgba(22, 20, 28, 0.75)", backdropFilter: "blur(12px)",
        border: `1px solid ${WARM(0.1)}`, borderRadius: 3,
        padding: "28px 32px", position: "relative",
        boxShadow: `0 4px 40px rgba(0,0,0,0.3), inset 0 1px 0 ${WARM(0.05)}`,
      }}>
        <CornerButton expanded={false} onClick={handleExpand} />

        {result.prev_text && (
          <p style={{ fontFamily: FONT, fontSize: 16, lineHeight: 1.75, color: DIM(0.25), margin: "0 0 20px 0", borderBottom: `1px solid ${WARM(0.06)}`, paddingBottom: 18 }}>
            {result.prev_text}
          </p>
        )}

        <p style={{ fontFamily: FONT, fontSize: 18.5, lineHeight: 1.8, color: PARCHMENT(0.92), margin: 0, letterSpacing: "0.01em", paddingRight: 40 }}>
          {result.paragraph_text}
        </p>

        {result.next_text && (
          <p style={{ fontFamily: FONT, fontSize: 16, lineHeight: 1.75, color: DIM(0.25), margin: "20px 0 0 0", borderTop: `1px solid ${WARM(0.06)}`, paddingTop: 18 }}>
            {result.next_text}
          </p>
        )}
      </div>

      {/* Bottom: collect */}
      <div style={{ display: "flex", gap: 20, marginTop: 10, paddingLeft: 2 }}>
        <button
          onClick={() => onCollect(result)}
          style={{
            background: "none", border: "none", fontFamily: FONT,
            fontStyle: "italic", fontSize: 13, color: WARM(0.4),
            cursor: "pointer", padding: "2px 0", transition: "color 0.3s ease",
          }}
          onMouseEnter={(e) => (e.target.style.color = WARM(0.8))}
          onMouseLeave={(e) => (e.target.style.color = WARM(0.4))}
        >
          + collect
        </button>
      </div>
    </div>
  );
}

// ─── Floating Card (lifted passage during transition) ────────────────────────

function FloatingCard({ result, rect, target, phase, highlightPos }) {
  if (!result || !rect) return null;

  const MOVE_MS = 400;
  const isAtTarget = target === "center";

  const prevTextHeight = result.prev_text ? 58 : 0;
  const originTop = rect.top + 28 + prevTextHeight;
  const originLeft = rect.left + 32;
  const originWidth = rect.width - 64;

  const targetTop = highlightPos ? highlightPos.top : window.innerHeight * 0.3;
  const targetLeft = highlightPos ? highlightPos.left : originLeft;
  const targetWidth = highlightPos ? highlightPos.width : originWidth;

  const currentTop = isAtTarget ? targetTop : originTop;
  const currentLeft = isAtTarget ? targetLeft : originLeft;
  const currentWidth = isAtTarget ? targetWidth : originWidth;

  let opacity = 1;
  if (phase === "book-enter") opacity = 0;

  return (
    <div style={{
      position: "fixed",
      top: currentTop,
      left: currentLeft,
      width: currentWidth,
      zIndex: 11,
      pointerEvents: "none",
      opacity,
      transition: `top ${MOVE_MS}ms cubic-bezier(0.4, 0, 0.2, 1), left ${MOVE_MS}ms cubic-bezier(0.4, 0, 0.2, 1), width ${MOVE_MS}ms cubic-bezier(0.4, 0, 0.2, 1), opacity 200ms ease`,
    }}>
      <p style={{
        fontFamily: FONT, fontSize: 18.5, lineHeight: 1.8,
        color: PARCHMENT(0.92), margin: 0, letterSpacing: "0.01em",
      }}>
        {result.paragraph_text}
      </p>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────

const CURTAIN_MS = 380;
const CARD_MOVE_MS = 400;
const CONTENT_MS = 300;
const CARD_BG = "rgba(22, 20, 28, 1)";

export default function GutemGrep() {
  const token = useInitialToken();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const [selectionTooltip, setSelectionTooltip] = useState({ text: null, position: null });
  const [collection, setCollection] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [expandedResult, setExpandedResult] = useState(null);
  const [viewPhase, setViewPhase] = useState("search");
  const [cardRect, setCardRect] = useState(null);
  const [curtainTarget, setCurtainTarget] = useState(null);
  const [floatingTarget, setFloatingTarget] = useState("origin");
  const [highlightPos, setHighlightPos] = useState(null);
  const [loadingToken, setLoadingToken] = useState(!!token);
  const timersRef = useRef([]);
  const inputRef = useRef(null);
  const epigraph = useRef(EPIGRAPHS[Math.floor(Math.random() * EPIGRAPHS.length)]);

  const clearTimers = () => { timersRef.current.forEach(clearTimeout); timersRef.current = []; };
  const later = (fn, ms) => { const t = setTimeout(fn, ms); timersRef.current.push(t); };

  // Load saved query if URL has a token
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    async function load() {
      try {
        const data = await getSavedQuery({ token });
        if (cancelled) return;
        if (data && data.results) {
          setResults(data.results);
          const reqBody = data.request_body ? JSON.parse(data.request_body) : null;
          setCurrentQuery(reqBody?.query || token);
          setQuery(reqBody?.query || "");
        }
      } catch {
        // If saved query fails, just show the search page
      } finally {
        if (!cancelled) setLoadingToken(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [token]);

  // ── EXPAND ──
  const doExpand = useCallback((result, rect) => {
    clearTimers();
    setSelectionTooltip({ text: null, position: null });
    setExpandedResult(result);
    setCardRect(rect || { top: window.innerHeight / 3, left: window.innerWidth / 4, width: window.innerWidth / 2, height: 200 });
    setFloatingTarget("origin");
    setHighlightPos(null);
    setCurtainTarget("card");
    setViewPhase("curtain-grow");

    requestAnimationFrame(() => requestAnimationFrame(() => setCurtainTarget("full")));

    later(() => {
      setViewPhase("card-settle");
      setFloatingTarget("center");

      later(() => {
        setViewPhase("book-enter");

        later(() => setViewPhase("book"), CONTENT_MS);
      }, CARD_MOVE_MS);
    }, CURTAIN_MS);
  }, []);

  // ── COLLAPSE (exact inverse) ──
  const doCollapse = useCallback(() => {
    clearTimers();
    setSelectionTooltip({ text: null, position: null });

    setFloatingTarget("center");
    setViewPhase("book-exit");

    later(() => {
      setViewPhase("card-return");
      setFloatingTarget("origin");

      later(() => {
        setViewPhase("curtain-shrink");
        setCurtainTarget("card");

        later(() => {
          setViewPhase("search");
          setExpandedResult(null);
          setCurtainTarget(null);
          setCardRect(null);
        }, CURTAIN_MS);
      }, CARD_MOVE_MS);
    }, CONTENT_MS);
  }, []);

  useEffect(() => () => clearTimers(), []);

  const doSearch = useCallback(async (searchText) => {
    if (!searchText.trim()) return;
    setSearching(true);
    setCurrentQuery(searchText);
    setQuery(searchText);
    setSelectionTooltip({ text: null, position: null });
    setExpandedResult(null);
    setViewPhase("search");
    setCurtainTarget(null);
    setCardRect(null);
    setFloatingTarget("origin");
    setHighlightPos(null);

    try {
      const resp = await postQuery({ query: searchText, topK: 10, includeContext: true });
      setResults(resp.results || []);
      // Update URL to saved query UUID without full page reload
      if (resp.uuid) {
        window.history.pushState({}, "", `/${resp.uuid}`);
      }
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }, []);

  useEffect(() => {
    const handler = () => {
      if (selectionTooltip.text) {
        setTimeout(() => {
          const sel = window.getSelection()?.toString().trim();
          if (!sel) setSelectionTooltip({ text: null, position: null });
        }, 200);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [selectionTooltip]);

  const handleTextSelect = (text, position) => {
    setSelectionTooltip({ text, position });
  };

  const handleNavigateToSimilar = (item) => {
    const existing = results ? results.find(r => r.embed_id === item.embed_id) : null;
    setExpandedResult(existing || {
      embed_id: item.embed_id, score: item.score,
      paragraph_text: item.paragraph_text, book_title: item.book_title,
      book_id: item.book_id || 0,
    });
  };

  const handleHighlightMeasured = useCallback((pos) => {
    setHighlightPos(pos);
  }, []);

  const showSearch = viewPhase === "search" || viewPhase === "curtain-grow" || viewPhase === "card-return" || viewPhase === "curtain-shrink";
  const showBook = ["curtain-grow", "card-settle", "book-enter", "book", "book-exit"].includes(viewPhase);
  const bookVisible = ["book-enter", "book", "book-exit"].includes(viewPhase);
  const showCurtain = !["search", "book"].includes(viewPhase);
  const showFloating = ["curtain-grow", "card-settle", "book-enter", "book-exit", "card-return", "curtain-shrink"].includes(viewPhase);
  const isHome = results === null && viewPhase === "search" && !loadingToken;

  const curtainStyle = (() => {
    if (!cardRect) return {};
    const isAtCard = curtainTarget === "card";
    return {
      position: "fixed",
      zIndex: 10,
      pointerEvents: "none",
      background: CARD_BG,
      border: `1px solid ${isAtCard ? WARM(0.1) : "transparent"}`,
      borderRadius: isAtCard ? 3 : 0,
      top: isAtCard ? cardRect.top : 0,
      left: isAtCard ? cardRect.left : 0,
      width: isAtCard ? cardRect.width : "100vw",
      height: isAtCard ? cardRect.height : "100vh",
      transition: `top ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), left ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), width ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), height ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), border-radius ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), border-color ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1), box-shadow ${CURTAIN_MS}ms cubic-bezier(0.4, 0, 0.2, 1)`,
      boxShadow: isAtCard ? `0 4px 40px rgba(0,0,0,0.3), inset 0 1px 0 ${WARM(0.05)}` : "none",
    };
  })();

  if (loadingToken) {
    return (
      <div style={{ minHeight: "100vh", background: "#060812", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <p style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 15, color: WARM(0.3), animation: "pulseGlow 1.5s ease infinite" }}>
          {"loading\u2026"}
        </p>
      </div>
    );
  }

  return (
    <div style={{ minHeight: "100vh", background: "#060812", position: "relative", overflow: "hidden" }}>
      <StarField />
      <Vignette />

      {/* ── CURTAIN ── */}
      {showCurtain && cardRect && <div style={curtainStyle} />}

      {/* ── FLOATING CARD ── */}
      {showFloating && expandedResult && cardRect && (
        <FloatingCard result={expandedResult} rect={cardRect} target={floatingTarget} phase={viewPhase} highlightPos={highlightPos} />
      )}

      <SelectionTooltip
        text={selectionTooltip.text} position={selectionTooltip.position}
        onSearch={(text) => doSearch(text)}
      />

      <CollectionSidebar
        items={collection} open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSearchAverage={() => {
          // Search using the text of collected passages concatenated
          const searchText = collection.map(c => c.paragraph_text).join(" ").slice(0, 500);
          doSearch(searchText);
          setSidebarOpen(false);
        }}
      />

      <div style={{ position: "relative", zIndex: 2, display: "flex", flexDirection: "column", alignItems: "center", minHeight: "100vh" }}>

        {/* ── SEARCH + RESULTS ── */}
        {showSearch && (
          <div style={{
            width: "100%", display: "flex", flexDirection: "column", alignItems: "center",
            pointerEvents: viewPhase === "search" ? "auto" : "none",
          }}>
            <header style={{
              width: "100%", maxWidth: 680, padding: "0 24px",
              marginTop: isHome ? "30vh" : 48,
              transition: "margin-top 0.6s ease", flexShrink: 0,
            }}>
              <div style={{ textAlign: "center", marginBottom: isHome ? 48 : 24, transition: "margin-bottom 0.5s ease" }}>
                <h1
                  style={{
                    fontFamily: FONT, fontWeight: 400,
                    fontSize: isHome ? 42 : 22,
                    letterSpacing: "0.06em", color: WARM(0.85),
                    margin: 0, transition: "font-size 0.5s ease",
                    cursor: results ? "pointer" : "default",
                  }}
                  onClick={() => {
                    if (results) {
                      setResults(null); setQuery(""); setCurrentQuery("");
                      window.history.pushState({}, "", "/");
                      setTimeout(() => inputRef.current?.focus(), 100);
                    }
                  }}
                >
                  gutemgrep
                </h1>

                {isHome && (
                  <p style={{
                    fontFamily: FONT, fontStyle: "italic", fontSize: 15,
                    color: WARM(0.3), marginTop: 16, lineHeight: 1.6,
                    maxWidth: 480, marginLeft: "auto", marginRight: "auto",
                    animation: "fadeIn 1.5s ease",
                  }}>
                    {epigraph.current.text}<br />
                    <span style={{ fontSize: 13, color: WARM(0.2) }}>{"\u2014"} {epigraph.current.author}</span>
                  </p>
                )}
              </div>

              <div>
                <div style={{ position: "relative" }}>
                  <input
                    ref={inputRef} type="text" value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); doSearch(query); } }}
                    placeholder={"search the library\u2026"}
                    autoFocus
                    style={{
                      width: "100%", background: "rgba(18, 16, 24, 0.6)",
                      border: `1px solid ${WARM(0.12)}`, borderRadius: 3,
                      padding: "14px 20px", fontFamily: FONT, fontStyle: "italic",
                      fontSize: 18, color: PARCHMENT(0.9), outline: "none",
                      transition: "border-color 0.3s ease, box-shadow 0.3s ease",
                      backdropFilter: "blur(8px)",
                    }}
                    onFocus={(e) => { e.target.style.borderColor = WARM(0.25); e.target.style.boxShadow = `0 0 40px ${WARM(0.04)}`; }}
                    onBlur={(e) => { e.target.style.borderColor = WARM(0.12); e.target.style.boxShadow = "none"; }}
                  />
                  {searching && (
                    <div style={{
                      position: "absolute", right: 18, top: "50%", transform: "translateY(-50%)",
                      fontFamily: FONT, fontStyle: "italic", fontSize: 13, color: WARM(0.4),
                      animation: "pulseGlow 1.5s ease infinite",
                    }}>
                      {"searching\u2026"}
                    </div>
                  )}
                </div>
              </div>

              {isHome && (
                <p style={{ textAlign: "center", fontFamily: FONT, fontSize: 13, color: WARM(0.2), marginTop: 16, fontStyle: "italic" }}>
                  semantic search over {">"}70,000 books
                </p>
              )}
            </header>

            {results && (
              <div style={{ width: "100%", maxWidth: 680, padding: "32px 24px 120px", animation: "fadeIn 0.4s ease" }}>
                <div style={{
                  display: "flex", justifyContent: "space-between", alignItems: "baseline",
                  marginBottom: 32, paddingBottom: 16, borderBottom: `1px solid ${WARM(0.06)}`,
                }}>
                  <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.35) }}>
                    {results.length} passages for {"\u201c"}{currentQuery}{"\u201d"}
                  </span>
                  <button
                    onClick={() => setSidebarOpen(true)}
                    style={{
                      background: "none", border: "none", fontFamily: FONT,
                      fontStyle: "italic", fontSize: 13, color: WARM(0.35),
                      cursor: "pointer", transition: "color 0.3s ease",
                    }}
                    onMouseEnter={(e) => (e.target.style.color = WARM(0.7))}
                    onMouseLeave={(e) => (e.target.style.color = WARM(0.35))}
                  >
                    collection ({collection.length})
                  </button>
                </div>

                {results.map((result, i) => (
                  <ResultCard
                    key={result.embed_id} result={result} index={i}
                    onTextSelect={handleTextSelect}
                    onExpand={(r, rect) => doExpand(r, rect)}
                    onCollect={(r) => {
                      if (!collection.find(c => c.embed_id === r.embed_id)) {
                        setCollection(prev => [...prev, r]);
                      }
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── BOOK READER ── */}
      {showBook && expandedResult && (
        <div style={{
          position: "fixed", inset: 0,
          display: "flex", flexDirection: "column", alignItems: "center",
          opacity: bookVisible ? 1 : 0,
          pointerEvents: bookVisible ? "auto" : "none",
          animation: viewPhase === "book-enter" ? `bookContentIn ${CONTENT_MS}ms cubic-bezier(0.16, 1, 0.3, 1) both` :
                     viewPhase === "book-exit" ? `bookContentOut ${CONTENT_MS}ms cubic-bezier(0.7, 0, 0.84, 0) both` : "none",
          zIndex: 12,
        }}>
          <BookReader
            key={expandedResult.embed_id}
            result={expandedResult}
            allResults={results || []}
            onCollapse={doCollapse}
            onTextSelect={handleTextSelect}
            onNavigateToSimilar={handleNavigateToSimilar}
            onHighlightMeasured={handleHighlightMeasured}
          />
        </div>
      )}
    </div>
  );
}
