#!/usr/bin/env python3
import csv
import re
import sys
import time
import urllib.request
import unicodedata

SOURCES = [
    {
        "id": "philosophers",
        "title": "list of philosophers",
        "url": "https://www.britannica.com/topic/list-of-philosophers-2027173",
    },
    {
        "id": "physicists",
        "title": "list of physicists",
        "url": "https://www.britannica.com/topic/list-of-physicists-2025130",
    },
    {
        "id": "chemists",
        "title": "list of chemists",
        "url": "https://www.britannica.com/topic/list-of-chemists-2028528",
    },
    {
        "id": "science_fiction_writers",
        "title": "list of science-fiction writers",
        "url": "https://www.britannica.com/art/list-of-science-fiction-writers",
    },
    {
        "id": "playwrights",
        "title": "list of playwrights",
        "url": "https://www.britannica.com/topic/list-of-playwrights-2030379",
    },
    {
        "id": "american_writers",
        "title": "list of American writers",
        "url": "https://www.britannica.com/topic/list-of-American-writers-2060492",
    },
]

OUT_PATH = "gutenberg/metadata/notable_authors_britannica.csv"


def _to_ascii(text: str) -> str:
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_name(text: str) -> str:
    text = _to_ascii(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _jina_url(url: str) -> str:
    stripped = url.replace("https://", "").replace("http://", "")
    return f"https://r.jina.ai/http://{stripped}"


def fetch_text(url: str) -> str:
    req = urllib.request.Request(
        _jina_url(url),
        headers={"User-Agent": "Mozilla/5.0 (compatible; gutemgrep/0.1)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_items(text: str):
    items = []
    section = None
    started = False
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):
            section = line[3:].strip()
            started = True
            continue
        if not started:
            continue
        match = re.match(r"^\s*[*-]\s+(.*)$", line)
        if not match:
            continue
        item = match.group(1).strip()
        if not item:
            continue
        items.append((item, section))
    return items


def main() -> int:
    aggregated = {}
    for source in SOURCES:
        text = fetch_text(source["url"])
        entries = extract_items(text)
        if not entries:
            print(f"warning: no entries parsed for {source['url']}", file=sys.stderr)
        for raw_name, section in entries:
            display = _to_ascii(raw_name).strip()
            if not display:
                continue
            norm = _normalize_name(display)
            if not norm:
                continue
            rec = aggregated.setdefault(
                norm,
                {
                    "name_display": display,
                    "sources": set(),
                    "urls": set(),
                    "sections": set(),
                },
            )
            rec["sources"].add(source["title"])
            rec["urls"].add(source["url"])
            if section:
                rec["sections"].add(section)
        time.sleep(0.2)

    rows = []
    for norm, rec in aggregated.items():
        rows.append(
            {
                "name_norm": norm,
                "name_display": rec["name_display"],
                "sources": "; ".join(sorted(rec["sources"])),
                "source_urls": "; ".join(sorted(rec["urls"])),
                "sections": "; ".join(sorted(rec["sections"])),
            }
        )

    rows.sort(key=lambda r: r["name_norm"])
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name_norm", "name_display", "sources", "source_urls", "sections"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
