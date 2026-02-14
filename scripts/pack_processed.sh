#!/usr/bin/env bash
set -euo pipefail

# Pack processed Gutenberg data (text/tokens/counts) into a tar.gz archive.
# Usage: ./scripts/pack_processed.sh [gutenberg_dir] [output_tar_gz]

GUTENBERG_DIR="${1:-/home/owner/dev/gutemgrep/gutenberg}"
OUT="${2:-processed_gutenberg.tar.gz}"

DATA_DIR="${GUTENBERG_DIR}/data"
TEXT_DIR="${DATA_DIR}/text"
TOKENS_DIR="${DATA_DIR}/tokens"
COUNTS_DIR="${DATA_DIR}/counts"

if [[ ! -d "$TEXT_DIR" || ! -d "$TOKENS_DIR" || ! -d "$COUNTS_DIR" ]]; then
  echo "Expected processed dirs missing under: $DATA_DIR" >&2
  exit 1
fi

tar -czf "$OUT" -C "$GUTENBERG_DIR" data/text data/tokens data/counts
echo "Wrote: $OUT"
