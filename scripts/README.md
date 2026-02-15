# Scripts

## add_notable_author.py

Adds a `notable_author` column to a Gutenberg-style `metadata.csv`, based on a list of notable authors.

Usage:

```bash
python scripts/add_notable_author.py \
  --metadata-csv gutenberg/metadata/metadata.csv \
  --authors-list authors_list_2.txt \
  --output-csv gutenberg/metadata/metadata_with_notable.csv
```

Notes on matching:
- Normalizes punctuation, case, and diacritics (e.g., Bronte -> Bronte).
- Handles "Last, First" to "First Last" reordering.
- Accepts parenthetical aliases when they look like names (e.g., "Saki (H. H. Munro)").
- For single-token notable names, matches the last token of the author name.
- Splits multiple authors only on `;` or `|` (conservative to avoid breaking institutional names).

Rationale / design:
- `metadata.csv` authors are often stored as "Last, First"; reordering is required for list matching.
- The notable list contains diacritics and punctuation variants, so normalization avoids false misses.
- Parentheticals are used for aliases but also for notes; a conservative heuristic reduces false positives.
- Single-token names (e.g., "Seneca") appear in `authors_list_2.txt` while metadata includes full names;
  matching on the last token captures common classical-name patterns with minimal complexity.
- Multi-author entries are rare and can be ambiguous; only `;` and `|` are split to avoid breaking
  institutional or corporate names that include "and" or "&".

Options:
- `--column-name`: change the output column name (default: `notable_author`).
- `--replace-column`: overwrite if the column already exists.

## Authors lists

Two lists are currently tracked at repo root:
- `authors_list.txt`: original shorter list.
- `authors_list_2.txt`: expanded list (superset) used for the latest matching runs.

The script accepts either list via `--authors-list`. Keep one name per line; parenthetical notes
are allowed, but only parentheticals that look like names are treated as aliases.
