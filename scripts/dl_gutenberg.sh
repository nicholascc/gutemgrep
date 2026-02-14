#!/usr/bin/env bash
set -euo pipefail

# Download the Project Gutenberg "main" collection via rsync.
# Default mirror is the Project Gutenberg high-speed rsync mirror (San Diego, USA).
DEST="${1:-data/gutenberg}"
MIRROR_URL="${GUTENBERG_MIRROR_URL:-rsync://gutenberg.pglaf.org/gutenberg}"

mkdir -p "$DEST"
rsync -av --delete "$MIRROR_URL" "$DEST"
