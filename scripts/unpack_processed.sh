#!/usr/bin/env bash
set -euo pipefail

# Unpack processed Gutenberg data tar.gz into a target Gutenberg dir.
# Usage: ./scripts/unpack_processed.sh <archive_tar_gz> [gutenberg_dir]

ARCHIVE="${1:-}"
GUTENBERG_DIR="${2:-/home/owner/dev/gutemgrep/gutenberg}"

if [[ -z "$ARCHIVE" ]]; then
  echo "Usage: $0 <archive_tar_gz> [gutenberg_dir]" >&2
  exit 1
fi

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 1
fi

mkdir -p "$GUTENBERG_DIR"
tar -xzf "$ARCHIVE" -C "$GUTENBERG_DIR"
echo "Unpacked into: $GUTENBERG_DIR/data/{text,tokens,counts}"
