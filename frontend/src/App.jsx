import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { postQuery, getSavedQuery, getBookByEmbed, createVector, listVectors, queryByVector, mutateText } from "./api.js";

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

// ─── Text Formatting (copy-safe) ─────────────────────────────────────────────
// Renders _underscores_ as italics and -- as em-dashes visually,
// but clipboard copies preserve the original _underscores_ and -- characters.
// Technique: original chars live in font-size:0 spans (invisible, but selected
// on copy); visual replacements use user-select:none (visible, skipped on copy).

const COPY_ONLY = { fontSize: 0, lineHeight: 0, overflow: "hidden" };
const DISPLAY_ONLY = { userSelect: "none", WebkitUserSelect: "none" };

function isWordChar(ch) {
  if (!ch) return false;
  const code = ch.charCodeAt(0);
  return (code >= 48 && code <= 57) || (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
}

function renderFormatted(text) {
  if (text == null) return null;
  const s = String(text);
  const parts = [];
  let last = 0;
  let i = 0;
  let k = 0;

  while (i < s.length) {
    // Check for -- (em-dash)
    if (s[i] === "-" && s[i + 1] === "-") {
      if (i > last) parts.push(s.slice(last, i));
      parts.push(
        <span key={`d${k++}`}>
          <span style={COPY_ONLY}>--</span>
          <span style={DISPLAY_ONLY}>{"\u2014"}</span>
        </span>
      );
      last = i + 2;
      i += 2;
      continue;
    }

    // Check for _italic_
    if (s[i] === "_") {
      const prev = i > 0 ? s[i - 1] : "";
      if (!isWordChar(prev)) {
        let j = i + 1;
        while (j < s.length) {
          if (s[j] === "_") {
            const next = j + 1 < s.length ? s[j + 1] : "";
            if (!isWordChar(next)) break;
          }
          j++;
        }
        if (j < s.length && s[j] === "_" && j > i + 1) {
          const content = s.slice(i + 1, j);
          if (content.trim().length > 0) {
            if (i > last) parts.push(s.slice(last, i));
            parts.push(
              <em key={`i${k++}`}>
                <span style={COPY_ONLY}>_</span>
                {content}
                <span style={COPY_ONLY}>_</span>
              </em>
            );
            last = j + 1;
            i = j + 1;
            continue;
          }
        }
      }
    }

    i++;
  }

  if (last < s.length) parts.push(s.slice(last));
  return parts.length > 0 ? parts : s;
}

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

function SimilarCarousel({ embedId, allResults, onNavigate, onCollect, collectedIds }) {
  const items = useMemo(() => {
    if (!allResults || allResults.length === 0) return [];
    return allResults
      .filter(x => x.embed_id !== embedId)
      .map(x => ({
        embed_id: x.embed_id, book_id: x.book_id,
        score: x.score,
        paragraph_text: x.paragraph_text,
        book_title: x.book_title,
        author: x.author,
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
          <SimilarCard
            key={item.embed_id} item={item} onClick={() => onNavigate(item)}
            onCollect={onCollect ? () => onCollect(item) : null}
            collected={collectedIds ? collectedIds.has(item.embed_id) : false}
          />
        ))}
      </div>
    </div>
  );
}

function SimilarCard({ item, onClick, onCollect, collected }) {
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
        {renderFormatted(item.paragraph_text.length > 150 ? item.paragraph_text.slice(0, 150) + "\u2026" : item.paragraph_text)}
      </p>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 12, color: WARM(0.45) }}>
          {renderFormatted(item.book_title)}
          {item.author && <span style={{ color: WARM(0.3), marginLeft: 4 }}>{"\u2014"} {item.author}</span>}
        </span>
        <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
          {onCollect && <CollectButton collected={collected} onCollect={onCollect} size="small" />}
          <span style={{ fontFamily: FONT, fontSize: 11, color: WARM(0.25), fontVariantNumeric: "tabular-nums" }}>
            {item.score.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

// ─── Book Reader (expanded mode) ─────────────────────────────────────────────

function BookReader({ result, allResults, onCollapse, onTextSelect, onNavigateToSimilar, onHighlightMeasured, onCollect, collectedIds }) {
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
            {renderFormatted(result.book_title)}
          </span>
          {result.author && (
            <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 15, color: WARM(0.45), marginLeft: 10 }}>
              {"\u2014"} {result.author}
            </span>
          )}
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
        {loaded && (
          <div style={{ animation: "fadeIn 0.5s ease" }}>
            {paragraphs.map((para) => {
              const isHighlight = para.embed_id === result.embed_id;
              return (
                <BookParagraphWithRef
                  key={para.embed_id} para={para} isHighlight={isHighlight}
                  innerRef={isHighlight ? highlightRef : null}
                  onCollect={onCollect ? (p) => onCollect({ embed_id: p.embed_id, paragraph_text: p.text, book_title: result.book_title, book_id: result.book_id }) : null}
                  collected={collectedIds ? collectedIds.has(para.embed_id) : false}
                />
              );
            })}
          </div>
        )}
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
          onCollect={onCollect}
          collectedIds={collectedIds}
        />
      </div>
    </div>
  );
}

const BookParagraphWithRef = ({ para, isHighlight, innerRef, onCollect, collected }) => {
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
        {renderFormatted(para.text)}
      </p>
      {hovered && onCollect && (
        <div style={{ marginTop: 4, opacity: 0.8 }}>
          <CollectButton collected={collected} onCollect={() => onCollect(para)} size="small" />
        </div>
      )}
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

// ─── localStorage helpers ────────────────────────────────────────────────────

const LS_COLLECTION_KEY = "gutemgrep_collection";
const LS_VECTORS_KEY = "gutemgrep_vector_ids";

function loadCollection() {
  try {
    const raw = localStorage.getItem(LS_COLLECTION_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveCollection(items) {
  try { localStorage.setItem(LS_COLLECTION_KEY, JSON.stringify(items)); } catch {}
}

function loadVectorIds() {
  try {
    const raw = localStorage.getItem(LS_VECTORS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveVectorIds(ids) {
  try { localStorage.setItem(LS_VECTORS_KEY, JSON.stringify(ids)); } catch {}
}

// ─── Collect Button (inline) ─────────────────────────────────────────────────

function CollectButton({ collected, onCollect, size = "normal" }) {
  const [hovered, setHovered] = useState(false);
  const fontSize = size === "small" ? 12 : 13;

  return (
    <button
      onClick={(e) => { e.stopPropagation(); if (!collected) onCollect(); }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: "none", border: "none", fontFamily: FONT,
        fontStyle: "italic", fontSize, padding: "2px 0",
        color: collected ? WARM(0.55) : WARM(hovered ? 0.8 : 0.4),
        cursor: collected ? "default" : "pointer",
        transition: "color 0.3s ease", whiteSpace: "nowrap",
      }}
    >
      {collected ? "\u2713 collected" : "+ collect"}
    </button>
  );
}

// ─── Alter Button (inline) ───────────────────────────────────────────────────

function AlterButton({ text, onAltered, size = "normal" }) {
  const [expanded, setExpanded] = useState(false);
  const [instruction, setInstruction] = useState("");
  const [loading, setLoading] = useState(false);
  const [hovered, setHovered] = useState(false);
  const inputRef = useRef(null);
  const fontSize = size === "small" ? 12 : 13;

  useEffect(() => {
    if (expanded && inputRef.current) inputRef.current.focus();
  }, [expanded]);

  const handleSubmit = async () => {
    if (!instruction.trim() || loading) return;
    setLoading(true);
    try {
      const resp = await mutateText({ text, instruction: instruction.trim() });
      if (resp.mutated_text) {
        onAltered(resp.mutated_text);
        setInstruction("");
        setExpanded(false);
      }
    } catch {
      // silently fail for now
    } finally {
      setLoading(false);
    }
  };

  if (expanded) {
    return (
      <div onClick={(e) => e.stopPropagation()} style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <input
          ref={inputRef}
          type="text"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSubmit();
            if (e.key === "Escape") { setExpanded(false); setInstruction(""); }
          }}
          placeholder={"how to alter\u2026"}
          disabled={loading}
          style={{
            background: "rgba(18, 16, 24, 0.6)",
            border: `1px solid ${WARM(0.15)}`, borderRadius: 3,
            padding: "4px 8px", fontFamily: FONT, fontStyle: "italic",
            fontSize: fontSize - 1, color: PARCHMENT(0.8), outline: "none",
            width: 150,
          }}
        />
        {loading ? (
          <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: fontSize - 1, color: WARM(0.4), animation: "pulseGlow 1.5s ease infinite", whiteSpace: "nowrap" }}>
            {"\u2026"}
          </span>
        ) : (
          <button
            onClick={handleSubmit}
            style={{
              background: "none", border: "none", fontFamily: FONT,
              fontStyle: "italic", fontSize: fontSize - 1, padding: "2px 0",
              color: WARM(instruction.trim() ? 0.6 : 0.25),
              cursor: instruction.trim() ? "pointer" : "default",
              whiteSpace: "nowrap",
            }}
          >
            go
          </button>
        )}
        <button
          onClick={() => { setExpanded(false); setInstruction(""); }}
          style={{
            background: "none", border: "none", fontFamily: FONT,
            fontSize: fontSize - 1, padding: "2px 0",
            color: WARM(0.3), cursor: "pointer", whiteSpace: "nowrap",
          }}
        >
          {"\u00d7"}
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={(e) => { e.stopPropagation(); setExpanded(true); }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: "none", border: "none", fontFamily: FONT,
        fontStyle: "italic", fontSize, padding: "2px 0",
        color: WARM(hovered ? 0.8 : 0.4),
        cursor: "pointer",
        transition: "color 0.3s ease", whiteSpace: "nowrap",
      }}
    >
      + alter
    </button>
  );
}

// ─── Sidebar Search Button ──────────────────────────────────────────────────

function SidebarSearchButton({ onClick }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: "none", border: "none", fontFamily: FONT,
        fontStyle: "italic", fontSize: 12, padding: "2px 0",
        color: WARM(hovered ? 0.8 : 0.4),
        cursor: "pointer",
        transition: "color 0.3s ease", whiteSpace: "nowrap",
      }}
    >
      search
    </button>
  );
}

// ─── Collection Sidebar (LEFT) ──────────────────────────────────────────────

function CollectionSidebar({ items, open, onClose, onRemove, selected, onToggleSelect, onCreateVector, customTexts, selectedCustomIndices, onToggleCustomSelect, onAddCustomText, onRemoveCustomText, onAlter, onSearchByText }) {
  const [newText, setNewText] = useState("");
  const [vectorName, setVectorName] = useState("");
  const [showNaming, setShowNaming] = useState(false);
  const [creating, setCreating] = useState(false);

  const hasSelection = selected.size > 0 || selectedCustomIndices.size > 0;

  const selectedCustomTexts = customTexts.filter((_, i) => selectedCustomIndices.has(i));

  const handleCreate = async () => {
    if (!vectorName.trim()) return;
    setCreating(true);
    try {
      await onCreateVector(vectorName.trim(), selected, selectedCustomTexts);
      setVectorName("");
      setShowNaming(false);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div style={{
      position: "fixed", left: 0, top: 0, bottom: 0, width: 360,
      background: "rgba(12, 10, 20, 0.95)", backdropFilter: "blur(20px)",
      borderRight: `1px solid ${WARM(0.08)}`, zIndex: 50,
      transform: open ? "translateX(0)" : "translateX(-100%)",
      transition: "transform 0.4s ease", padding: "40px 28px",
      display: "flex", flexDirection: "column",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 24, flexShrink: 0 }}>
        <h3 style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 18, color: WARM(0.8), margin: 0, fontWeight: 400 }}>
          collected passages
        </h3>
        <button onClick={onClose} style={{ background: "none", border: "none", fontFamily: FONT, fontSize: 14, color: WARM(0.4), cursor: "pointer" }}>
          close
        </button>
      </div>

      <div style={{ flex: 1, overflowY: "auto", marginBottom: 16 }}>
        {items.length === 0 && customTexts.length === 0 ? (
          <p style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.3), lineHeight: 1.7 }}>
            {"Click \u201c+ collect\u201d on any passage to save it here. Select passages and create an averaged embedding vector for similarity search."}
          </p>
        ) : (
          <>
            {items.map((item) => {
              const isSelected = selected.has(item.embed_id);
              return (
                <div key={item.embed_id} style={{
                  display: "flex", gap: 10, padding: "12px 0",
                  borderBottom: `1px solid ${WARM(0.06)}`, alignItems: "flex-start",
                }}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggleSelect(item.embed_id)}
                    style={{ marginTop: 4, flexShrink: 0 }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontFamily: FONT, fontSize: 13.5, lineHeight: 1.6, color: DIM(isSelected ? 0.85 : 0.65), margin: "0 0 4px 0", transition: "color 0.2s ease" }}>
                      {renderFormatted(item.paragraph_text.length > 100 ? item.paragraph_text.slice(0, 100) + "\u2026" : item.paragraph_text)}
                    </p>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                      <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 11.5, color: WARM(0.3) }}>{renderFormatted(item.book_title)}</span>
                      <span style={{ color: WARM(0.1) }}>{"\u00b7"}</span>
                      <AlterButton text={item.paragraph_text} onAltered={onAlter} size="small" />
                      <SidebarSearchButton onClick={() => onSearchByText(item.paragraph_text)} />
                    </div>
                  </div>
                  <button
                    onClick={() => onRemove(item.embed_id)}
                    style={{
                      background: "none", border: "none", color: WARM(0.25),
                      cursor: "pointer", fontSize: 16, lineHeight: 1, padding: "2px 4px",
                      flexShrink: 0, transition: "color 0.2s ease",
                    }}
                    onMouseEnter={(e) => (e.target.style.color = WARM(0.6))}
                    onMouseLeave={(e) => (e.target.style.color = WARM(0.25))}
                    title="Remove from collection"
                  >
                    {"\u00d7"}
                  </button>
                </div>
              );
            })}

            {customTexts.map((text, i) => {
              const isSelected = selectedCustomIndices.has(i);
              return (
                <div key={`custom-${i}`} style={{
                  display: "flex", gap: 10, padding: "12px 0",
                  borderBottom: `1px solid ${WARM(0.06)}`, alignItems: "flex-start",
                }}>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggleCustomSelect(i)}
                    style={{ marginTop: 4, flexShrink: 0 }}
                  />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontFamily: FONT, fontSize: 13.5, lineHeight: 1.6, color: DIM(isSelected ? 0.85 : 0.65), margin: 0, fontStyle: "italic", transition: "color 0.2s ease" }}>
                      {text.length > 100 ? text.slice(0, 100) + "\u2026" : text}
                    </p>
                    <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                      <span style={{ fontFamily: FONT, fontSize: 11.5, color: WARM(0.3), fontStyle: "italic" }}>
                        <span style={{ fontFamily: FONT, fontSize: 10, color: WARM(0.35), marginRight: 4, fontStyle: "normal" }}>T</span>
                        custom text
                      </span>
                      <span style={{ color: WARM(0.1) }}>{"\u00b7"}</span>
                      <AlterButton text={text} onAltered={onAlter} size="small" />
                      <SidebarSearchButton onClick={() => onSearchByText(text)} />
                    </div>
                  </div>
                  <button
                    onClick={() => onRemoveCustomText(i)}
                    style={{
                      background: "none", border: "none", color: WARM(0.25),
                      cursor: "pointer", fontSize: 16, lineHeight: 1, padding: "2px 4px",
                      flexShrink: 0, transition: "color 0.2s ease",
                    }}
                    onMouseEnter={(e) => (e.target.style.color = WARM(0.6))}
                    onMouseLeave={(e) => (e.target.style.color = WARM(0.25))}
                  >
                    {"\u00d7"}
                  </button>
                </div>
              );
            })}
          </>
        )}
      </div>

      {/* Custom text input */}
      <div style={{ flexShrink: 0, borderTop: `1px solid ${WARM(0.06)}`, paddingTop: 14 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <input
            type="text"
            value={newText}
            onChange={(e) => setNewText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && newText.trim()) {
                onAddCustomText(newText.trim());
                setNewText("");
              }
            }}
            placeholder={"add custom text\u2026"}
            style={{
              flex: 1, background: "rgba(18, 16, 24, 0.6)",
              border: `1px solid ${WARM(0.1)}`, borderRadius: 3,
              padding: "8px 12px", fontFamily: FONT, fontStyle: "italic",
              fontSize: 13, color: PARCHMENT(0.8), outline: "none",
            }}
          />
          <button
            onClick={() => { if (newText.trim()) { onAddCustomText(newText.trim()); setNewText(""); } }}
            style={{
              background: WARM(0.06), border: `1px solid ${WARM(0.1)}`, borderRadius: 3,
              padding: "8px 12px", fontFamily: FONT, fontSize: 13, color: WARM(0.5),
              cursor: "pointer", flexShrink: 0,
            }}
          >
            add
          </button>
        </div>

        {/* Create vector controls */}
        {hasSelection && !showNaming && (
          <button
            onClick={() => setShowNaming(true)}
            style={{
              width: "100%", background: WARM(0.08),
              border: `1px solid ${WARM(0.15)}`, borderRadius: 3,
              padding: "10px 16px", fontFamily: FONT, fontStyle: "italic",
              fontSize: 14, color: WARM(0.7), cursor: "pointer", transition: "all 0.3s ease",
            }}
            onMouseEnter={(e) => { e.target.style.background = WARM(0.12); e.target.style.borderColor = WARM(0.25); }}
            onMouseLeave={(e) => { e.target.style.background = WARM(0.08); e.target.style.borderColor = WARM(0.15); }}
          >
            create vector ({selected.size} passage{selected.size !== 1 ? "s" : ""}{selectedCustomIndices.size > 0 ? ` + ${selectedCustomIndices.size} text` : ""})
          </button>
        )}

        {showNaming && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <input
              type="text"
              value={vectorName}
              onChange={(e) => setVectorName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); }}
              placeholder={"name this vector\u2026"}
              autoFocus
              style={{
                width: "100%", background: "rgba(18, 16, 24, 0.6)",
                border: `1px solid ${WARM(0.15)}`, borderRadius: 3,
                padding: "10px 14px", fontFamily: FONT, fontStyle: "italic",
                fontSize: 14, color: PARCHMENT(0.85), outline: "none",
              }}
            />
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={handleCreate}
                disabled={creating || !vectorName.trim()}
                style={{
                  flex: 1, background: WARM(creating ? 0.04 : 0.1),
                  border: `1px solid ${WARM(0.2)}`, borderRadius: 3,
                  padding: "8px 12px", fontFamily: FONT, fontStyle: "italic",
                  fontSize: 13, color: WARM(creating ? 0.4 : 0.7),
                  cursor: creating ? "wait" : "pointer",
                }}
              >
                {creating ? "creating\u2026" : "create"}
              </button>
              <button
                onClick={() => { setShowNaming(false); setVectorName(""); }}
                style={{
                  background: "none", border: `1px solid ${WARM(0.1)}`, borderRadius: 3,
                  padding: "8px 12px", fontFamily: FONT, fontSize: 13, color: WARM(0.4),
                  cursor: "pointer",
                }}
              >
                cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Collection Tab (closed indicator) ──────────────────────────────────────

function CollectionTab({ count, onClick }) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "fixed", left: 0, top: "50%", transform: "translateY(-50%)",
        zIndex: 49, background: hovered ? "rgba(12, 10, 20, 0.95)" : "rgba(12, 10, 20, 0.85)",
        border: `1px solid ${WARM(hovered ? 0.2 : 0.1)}`, borderLeft: "none",
        borderRadius: "0 6px 6px 0", padding: "16px 10px",
        cursor: "pointer", backdropFilter: "blur(12px)",
        transition: "all 0.3s ease", display: "flex", flexDirection: "column",
        alignItems: "center", gap: 6,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <rect x="2" y="2" width="12" height="12" rx="1.5" stroke={WARM(hovered ? 0.7 : 0.45)} strokeWidth="1.5" fill="none" />
        <line x1="5" y1="5.5" x2="11" y2="5.5" stroke={WARM(hovered ? 0.5 : 0.3)} strokeWidth="1" />
        <line x1="5" y1="8" x2="11" y2="8" stroke={WARM(hovered ? 0.5 : 0.3)} strokeWidth="1" />
        <line x1="5" y1="10.5" x2="9" y2="10.5" stroke={WARM(hovered ? 0.5 : 0.3)} strokeWidth="1" />
      </svg>
      {count > 0 && (
        <span style={{
          fontFamily: FONT, fontSize: 11, color: WARM(hovered ? 0.7 : 0.45),
          fontVariantNumeric: "tabular-nums",
        }}>
          {count}
        </span>
      )}
    </button>
  );
}

// ─── Vectors Sidebar (RIGHT) ────────────────────────────────────────────────

function VectorsSidebar({ vectors, open, onClose, onSearch, onDelete }) {
  return (
    <div style={{
      position: "fixed", right: 0, top: 0, bottom: 0, width: 340,
      background: "rgba(12, 10, 20, 0.95)", backdropFilter: "blur(20px)",
      borderLeft: `1px solid ${WARM(0.08)}`, zIndex: 50,
      transform: open ? "translateX(0)" : "translateX(100%)",
      transition: "transform 0.4s ease", padding: "40px 28px",
      display: "flex", flexDirection: "column",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 24, flexShrink: 0 }}>
        <h3 style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 18, color: WARM(0.8), margin: 0, fontWeight: 400 }}>
          saved vectors
        </h3>
        <button onClick={onClose} style={{ background: "none", border: "none", fontFamily: FONT, fontSize: 14, color: WARM(0.4), cursor: "pointer" }}>
          close
        </button>
      </div>

      <div style={{ flex: 1, overflowY: "auto" }}>
        {vectors.length === 0 ? (
          <p style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.3), lineHeight: 1.7 }}>
            {"Create vectors from your collected passages. Each vector is an averaged embedding you can search by."}
          </p>
        ) : (
          vectors.map((v) => (
            <VectorCard key={v.id} vector={v} onSearch={() => onSearch(v)} onDelete={() => onDelete(v.id)} />
          ))
        )}
      </div>
    </div>
  );
}

function VectorCard({ vector, onSearch, onDelete }) {
  const [hovered, setHovered] = useState(false);
  const sourceCount = (vector.source_embed_ids?.length || 0) + (vector.source_texts?.length || 0);

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: "14px 0", borderBottom: `1px solid ${WARM(0.06)}`,
        transition: "background 0.2s ease",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontFamily: FONT, fontSize: 15, color: WARM(hovered ? 0.85 : 0.7), transition: "color 0.2s ease" }}>
          {vector.name}
        </span>
        <button
          onClick={onDelete}
          style={{
            background: "none", border: "none", color: WARM(0.2),
            cursor: "pointer", fontSize: 14, padding: "0 4px",
            transition: "color 0.2s ease",
          }}
          onMouseEnter={(e) => (e.target.style.color = WARM(0.5))}
          onMouseLeave={(e) => (e.target.style.color = WARM(0.2))}
        >
          {"\u00d7"}
        </button>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 12, color: WARM(0.3) }}>
          {sourceCount} source{sourceCount !== 1 ? "s" : ""}
        </span>
        <button
          onClick={onSearch}
          style={{
            background: WARM(hovered ? 0.1 : 0.06), border: `1px solid ${WARM(hovered ? 0.2 : 0.1)}`,
            borderRadius: 3, padding: "5px 14px", fontFamily: FONT, fontStyle: "italic",
            fontSize: 12, color: WARM(hovered ? 0.7 : 0.5), cursor: "pointer",
            transition: "all 0.25s ease",
          }}
        >
          search
        </button>
      </div>
    </div>
  );
}

function VectorsTab({ count, onClick }) {
  const [hovered, setHovered] = useState(false);

  if (count === 0) return null;

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "fixed", right: 0, top: "50%", transform: "translateY(-50%)",
        zIndex: 49, background: hovered ? "rgba(12, 10, 20, 0.95)" : "rgba(12, 10, 20, 0.85)",
        border: `1px solid ${WARM(hovered ? 0.2 : 0.1)}`, borderRight: "none",
        borderRadius: "6px 0 0 6px", padding: "16px 10px",
        cursor: "pointer", backdropFilter: "blur(12px)",
        transition: "all 0.3s ease", display: "flex", flexDirection: "column",
        alignItems: "center", gap: 6,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <circle cx="8" cy="8" r="5.5" stroke={WARM(hovered ? 0.7 : 0.45)} strokeWidth="1.5" fill="none" />
        <circle cx="8" cy="8" r="2" fill={WARM(hovered ? 0.5 : 0.3)} />
      </svg>
      <span style={{
        fontFamily: FONT, fontSize: 11, color: WARM(hovered ? 0.7 : 0.45),
        fontVariantNumeric: "tabular-nums",
      }}>
        {count}
      </span>
    </button>
  );
}

// ─── Result Card ─────────────────────────────────────────────────────────────

function ResultCard({ result, index, onTextSelect, onExpand, onCollect, collected, onAlter }) {
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
      {/* Book title + author + score */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10, padding: "0 2px" }}>
        <span style={{ fontFamily: FONT, fontStyle: "italic", fontSize: 14, color: WARM(0.7), letterSpacing: "0.02em" }}>
          {renderFormatted(result.book_title)}
          {result.author && <span style={{ color: WARM(0.4), marginLeft: 6, fontSize: 13 }}>{"\u2014"} {result.author}</span>}
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
            {renderFormatted(result.prev_text)}
          </p>
        )}

        <p style={{ fontFamily: FONT, fontSize: 18.5, lineHeight: 1.8, color: PARCHMENT(0.92), margin: 0, letterSpacing: "0.01em", paddingRight: 40 }}>
          {renderFormatted(result.paragraph_text)}
        </p>

        {result.next_text && (
          <p style={{ fontFamily: FONT, fontSize: 16, lineHeight: 1.75, color: DIM(0.25), margin: "20px 0 0 0", borderTop: `1px solid ${WARM(0.06)}`, paddingTop: 18 }}>
            {renderFormatted(result.next_text)}
          </p>
        )}
      </div>

      {/* Bottom: collect + alter */}
      <div style={{ display: "flex", gap: 20, marginTop: 10, paddingLeft: 2 }}>
        <CollectButton collected={collected} onCollect={() => onCollect(result)} />
        <AlterButton text={result.paragraph_text} onAltered={(mutated) => onAlter(mutated)} />
      </div>
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────────────────────

const FADE_MS = 200;

export default function GutemGrep() {
  const token = useInitialToken();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");
  const [activeVectorSearch, setActiveVectorSearch] = useState(null); // { id, name } when searching by vector
  const [selectionTooltip, setSelectionTooltip] = useState({ text: null, position: null });
  const [collection, setCollection] = useState(() => loadCollection());
  const [customTexts, setCustomTexts] = useState([]);
  const [selectedCustomIndices, setSelectedCustomIndices] = useState(new Set());
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [collectionOpen, setCollectionOpen] = useState(false);
  const [vectors, setVectors] = useState([]);
  const [vectorsOpen, setVectorsOpen] = useState(false);
  const [expandedResult, setExpandedResult] = useState(null);
  const [viewPhase, setViewPhase] = useState("search"); // "search" | "fade-out" | "book" | "fade-out-book"
  const [loadingToken, setLoadingToken] = useState(!!token);
  const timersRef = useRef([]);
  const inputRef = useRef(null);
  const epigraph = useRef(EPIGRAPHS[Math.floor(Math.random() * EPIGRAPHS.length)]);

  // Derived: set of collected embed_ids for O(1) lookup
  const collectedIds = useMemo(() => new Set(collection.map(c => c.embed_id)), [collection]);

  const clearTimers = () => { timersRef.current.forEach(clearTimeout); timersRef.current = []; };
  const later = (fn, ms) => { const t = setTimeout(fn, ms); timersRef.current.push(t); };

  // Force plain-text-only copy so the clipboard gets _underscores_ and --
  // instead of HTML with <em> tags and em-dash characters.
  useEffect(() => {
    const handler = (e) => {
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) return;
      e.preventDefault();
      e.clipboardData.setData("text/plain", selection.toString());
    };
    document.addEventListener("copy", handler);
    return () => document.removeEventListener("copy", handler);
  }, []);

  // Persist collection to localStorage
  useEffect(() => { saveCollection(collection); }, [collection]);

  // Load saved vectors from localStorage on mount
  useEffect(() => {
    const ids = loadVectorIds();
    if (ids.length > 0) {
      listVectors({ ids }).then(data => {
        if (data?.vectors) setVectors(data.vectors);
      }).catch(() => {});
    }
  }, []);

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
  const doExpand = useCallback((result, _rect) => {
    clearTimers();
    setSelectionTooltip({ text: null, position: null });
    setExpandedResult(result);
    setViewPhase("fade-out");
    later(() => setViewPhase("book"), FADE_MS);
  }, []);

  // ── COLLAPSE ──
  const doCollapse = useCallback(() => {
    clearTimers();
    setSelectionTooltip({ text: null, position: null });
    setViewPhase("fade-out-book");
    later(() => {
      setViewPhase("search");
      setExpandedResult(null);
    }, FADE_MS);
  }, []);

  useEffect(() => () => clearTimers(), []);

  // Lock body scroll when book reader is open to prevent scrollbar jolt
  useEffect(() => {
    const inBook = viewPhase === "book" || viewPhase === "fade-out";
    if (inBook) {
      const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
      document.body.style.overflow = "hidden";
      document.body.style.paddingRight = `${scrollbarWidth}px`;
    } else {
      document.body.style.overflow = "";
      document.body.style.paddingRight = "";
    }
    return () => {
      document.body.style.overflow = "";
      document.body.style.paddingRight = "";
    };
  }, [viewPhase]);

  const doSearch = useCallback(async (searchText) => {
    if (!searchText.trim()) return;
    setSearching(true);
    setCurrentQuery(searchText);
    setQuery(searchText);
    setActiveVectorSearch(null);
    setSelectionTooltip({ text: null, position: null });
    setExpandedResult(null);
    setViewPhase("search");

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

  const doSearchByVector = useCallback(async (vector) => {
    setSearching(true);
    setCurrentQuery(vector.name);
    setActiveVectorSearch({ id: vector.id, name: vector.name });
    setQuery("");
    setSelectionTooltip({ text: null, position: null });
    setExpandedResult(null);
    setViewPhase("search");
    setVectorsOpen(false);

    try {
      const resp = await queryByVector({ vectorId: vector.id, topK: 10, includeContext: true });
      setResults(resp.results || []);
      if (resp.uuid) {
        window.history.pushState({}, "", `/${resp.uuid}`);
      }
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }, []);

  // ── Collection handlers ──
  const handleCollect = useCallback((item) => {
    setCollection(prev => {
      if (prev.find(c => c.embed_id === item.embed_id)) return prev;
      return [...prev, { embed_id: item.embed_id, paragraph_text: item.paragraph_text, book_title: item.book_title, book_id: item.book_id }];
    });
  }, []);

  const handleRemoveFromCollection = useCallback((embedId) => {
    setCollection(prev => prev.filter(c => c.embed_id !== embedId));
    setSelectedIds(prev => { const next = new Set(prev); next.delete(embedId); return next; });
  }, []);

  const handleToggleSelect = useCallback((embedId) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(embedId)) next.delete(embedId);
      else next.add(embedId);
      return next;
    });
  }, []);

  const handleToggleCustomSelect = useCallback((index) => {
    setSelectedCustomIndices(prev => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const handleAddCustomText = useCallback((text) => {
    setCustomTexts(prev => {
      const next = [...prev, text];
      // Auto-select newly added custom text
      setSelectedCustomIndices(prevSel => new Set([...prevSel, next.length - 1]));
      return next;
    });
  }, []);

  const handleRemoveCustomText = useCallback((index) => {
    setCustomTexts(prev => prev.filter((_, i) => i !== index));
    setSelectedCustomIndices(prev => {
      const next = new Set();
      for (const idx of prev) {
        if (idx < index) next.add(idx);
        else if (idx > index) next.add(idx - 1);
      }
      return next;
    });
  }, []);

  const handleCreateVector = useCallback(async (name, selectedEmbedIds, texts) => {
    const embedIds = Array.from(selectedEmbedIds);
    const resp = await createVector({ name, embedIds, customTexts: texts });
    const newVector = { id: resp.id, name: resp.name, source_embed_ids: resp.source_embed_ids, source_texts: resp.source_texts, created_at: resp.created_at };
    setVectors(prev => {
      const next = [...prev, newVector];
      saveVectorIds(next.map(v => v.id));
      return next;
    });
    // Clear selections after creating
    setSelectedIds(new Set());
    setCustomTexts([]);
    setSelectedCustomIndices(new Set());
    // Open vectors sidebar to show the new vector
    setVectorsOpen(true);
    setCollectionOpen(false);
  }, []);

  const handleDeleteVector = useCallback((vectorId) => {
    setVectors(prev => {
      const next = prev.filter(v => v.id !== vectorId);
      saveVectorIds(next.map(v => v.id));
      return next;
    });
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

  const showSearch = viewPhase === "search" || viewPhase === "fade-out";
  const showBook = viewPhase === "book" || viewPhase === "fade-out-book";
  const isHome = results === null && viewPhase === "search" && !loadingToken;

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

      <SelectionTooltip
        text={selectionTooltip.text} position={selectionTooltip.position}
        onSearch={(text) => doSearch(text)}
      />

      {/* ── COLLECTION SIDEBAR (LEFT) ── */}
      <CollectionSidebar
        items={collection} open={collectionOpen}
        onClose={() => setCollectionOpen(false)}
        onRemove={handleRemoveFromCollection}
        selected={selectedIds}
        onToggleSelect={handleToggleSelect}
        onCreateVector={handleCreateVector}
        customTexts={customTexts}
        selectedCustomIndices={selectedCustomIndices}
        onToggleCustomSelect={handleToggleCustomSelect}
        onAddCustomText={handleAddCustomText}
        onRemoveCustomText={handleRemoveCustomText}
        onAlter={handleAddCustomText}
        onSearchByText={(text) => { setCollectionOpen(false); doSearch(text); }}
      />
      {!collectionOpen && <CollectionTab count={collection.length} onClick={() => setCollectionOpen(true)} />}

      {/* ── VECTORS SIDEBAR (RIGHT) ── */}
      <VectorsSidebar
        vectors={vectors} open={vectorsOpen}
        onClose={() => setVectorsOpen(false)}
        onSearch={doSearchByVector}
        onDelete={handleDeleteVector}
      />
      {!vectorsOpen && <VectorsTab count={vectors.length} onClick={() => setVectorsOpen(true)} />}

      <div style={{ position: "relative", zIndex: 2, display: "flex", flexDirection: "column", alignItems: "center", width: "100%", minHeight: "100vh" }}>

        {/* ── SEARCH + RESULTS ── */}
        {showSearch && (
          <div style={{
            width: "100%", display: "flex", flexDirection: "column", alignItems: "center",
            pointerEvents: viewPhase === "search" ? "auto" : "none",
            opacity: viewPhase === "fade-out" ? 0 : 1,
            transition: `opacity ${FADE_MS}ms ease`,
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
                  in quire
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
                    {results.length} passages {activeVectorSearch
                      ? <>{"via vector \u201c"}{activeVectorSearch.name}{"\u201d"}</>
                      : <>{" for \u201c"}{currentQuery}{"\u201d"}</>
                    }
                  </span>
                </div>

                {results.map((result, i) => (
                  <ResultCard
                    key={result.embed_id} result={result} index={i}
                    onTextSelect={handleTextSelect}
                    onExpand={(r, rect) => doExpand(r, rect)}
                    onCollect={handleCollect}
                    collected={collectedIds.has(result.embed_id)}
                    onAlter={handleAddCustomText}
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
          opacity: viewPhase === "fade-out-book" ? 0 : 1,
          pointerEvents: viewPhase === "book" ? "auto" : "none",
          transition: `opacity ${FADE_MS}ms ease`,
          zIndex: 12,
        }}>
          <BookReader
            key={expandedResult.embed_id}
            result={expandedResult}
            allResults={results || []}
            onCollapse={doCollapse}
            onTextSelect={handleTextSelect}
            onNavigateToSimilar={handleNavigateToSimilar}
            onCollect={handleCollect}
            collectedIds={collectedIds}
          />
        </div>
      )}
    </div>
  );
}
