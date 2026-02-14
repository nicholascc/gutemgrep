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
