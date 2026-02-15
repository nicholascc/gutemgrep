#!/usr/bin/env python3
import argparse
import glob
import os
import statistics
import sys


def _iter_text_files(inputs: list[str]) -> list[str]:
    paths: list[str] = []
    for item in inputs:
        if any(ch in item for ch in "*?[]"):
            paths.extend(glob.glob(item))
            continue
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "*_text.txt")))
            continue
        paths.append(item)

    out: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
    return sorted(set(out))


def _get_encoding(*, model: str | None, encoding: str):
    try:
        import tiktoken  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "tiktoken is not installed.\n"
            "Install it (in a venv) and re-run, e.g.:\n"
            "  python -m venv .venv && . .venv/bin/activate && python -m pip install tiktoken"
        ) from e

    if model:
        return tiktoken.encoding_for_model(model)
    return tiktoken.get_encoding(encoding)


def _count_tokens(path: str, enc) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return len(enc.encode_ordinary(text))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Count tokens in Project Gutenberg text files using tiktoken."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Files/dirs/globs. Dirs scan for '*_text.txt'. Default: data/text/*_text.txt",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name (ignored if --model is set). Default: cl100k_base",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for tiktoken.encoding_for_model(...), e.g. gpt-4o-mini.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N files (after sorting).",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print one line per file: <path>\\t<count>.",
    )
    args = parser.parse_args(argv)

    inputs = args.inputs or ["data/text/*_text.txt"]
    paths = _iter_text_files(inputs)
    if args.limit is not None:
        paths = paths[: args.limit]

    if not paths:
        print("No input files found.", file=sys.stderr)
        return 2

    enc = _get_encoding(model=args.model, encoding=args.encoding)

    counts: list[int] = []
    min_item: tuple[str, int] | None = None
    max_item: tuple[str, int] | None = None

    for p in paths:
        c = _count_tokens(p, enc)
        counts.append(c)
        if args.per_file:
            print(f"{p}\t{c}")
        if min_item is None or c < min_item[1]:
            min_item = (p, c)
        if max_item is None or c > max_item[1]:
            max_item = (p, c)

    if len(counts) == 1:
        if not args.per_file:
            print(counts[0])
        return 0

    counts_sorted = sorted(counts)
    n = len(counts_sorted)

    def pct(p: int) -> int:
        k = int(round((p / 100) * (n - 1)))
        return counts_sorted[k]

    print(f"files\t{n}")
    print(f"total\t{sum(counts_sorted)}")
    print(f"mean\t{sum(counts_sorted)/n:.1f}")
    print(f"median\t{statistics.median(counts_sorted)}")
    print(f"p10\t{pct(10)}")
    print(f"p90\t{pct(90)}")
    if min_item:
        print(f"min\t{min_item[1]}\t{min_item[0]}")
    if max_item:
        print(f"max\t{max_item[1]}\t{max_item[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

