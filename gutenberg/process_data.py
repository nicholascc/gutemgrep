"""
Process counts for all PG data.

Written by
M. Gerlach and F. Font-Clos

"""
import os
from os.path import join
import argparse
import glob
import ast
import pandas as pd
import time
import traceback
import concurrent.futures as cf

from src.pipeline import process_book
from src.utils import get_langs_dict

def _outputs_exist(pg_number, text_dir, tokens_dir, counts_dir):
    return (
        os.path.isfile(os.path.join(text_dir, "PG%s_text.txt" % pg_number)) and
        os.path.isfile(os.path.join(tokens_dir, "PG%s_tokens.txt" % pg_number)) and
        os.path.isfile(os.path.join(counts_dir, "PG%s_counts.txt" % pg_number))
    )


def _process_single(filename, language, text_dir, tokens_dir, counts_dir, log_file, per_process_log, needed):
    try:
        if needed:
            if per_process_log and log_file != "":
                log_path = "%s.%d" % (log_file, os.getpid())
            else:
                log_path = log_file
            process_book(
                path_to_raw_file=filename,
                text_dir=text_dir,
                tokens_dir=tokens_dir,
                counts_dir=counts_dir,
                language=language,
                log_file=log_path
            )
        return (filename, None, None, needed)
    except UnicodeDecodeError:
        return (filename, "unicode", None, needed)
    except Exception:
        return (filename, "exception", traceback.format_exc(), needed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Processing raw texts from Project Gutenberg:"
        " i) removing headers,ii) tokenizing, and iii) counting words.")
    # raw folder
    parser.add_argument(
        "-r", "--raw",
        help="Path to the raw-folder",
        default='data/raw/',
        type=str)
    # text folder
    parser.add_argument(
        "-ote", "--output_text",
        help="Path to text-output (text_dir)",
        default='data/text/',
        type=str)
    # tokens folder
    parser.add_argument(
        "-oto", "--output_tokens",
        help="Path to tokens-output (tokens_dir)",
        default='data/tokens/',
        type=str)
    # counts folder
    parser.add_argument(
        "-oco", "--output_counts",
        help="Path to counts-output (counts_dir)",
        default='data/counts/',
        type=str)
    # pattern to specify subset of books
    parser.add_argument(
        "-p", "--pattern",
        help="Patttern to specify a subset of books",
        default='*',
        type=str)
    # skip recently modified raw files (avoid partial rsync writes)
    parser.add_argument(
        "--min_age_seconds",
        help="Skip raw files modified within the last N seconds",
        default=0,
        type=int)

    # quiet argument, to supress info
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode, do not print info, warnings, etc"
    )
    # workers
    parser.add_argument(
        "-w", "--workers",
        help="Number of worker processes (default: CPU count)",
        default=os.cpu_count() or 1,
        type=int
    )

    # log file
    parser.add_argument(
        "-l", "--log_file",
        help="Path to log file",
        default=".log",
        type=str)

    # add arguments to parser
    args = parser.parse_args()

    # check whether the out-put directories exist
    if os.path.isdir(args.output_text) is False:
        raise ValueError("The directory for output of texts '%s' "
                         "does not exist" % (args.output_text))
    if os.path.isdir(args.output_tokens) is False:
        raise ValueError("The directory for output of tokens '%s' "
                         "does not exist" % (args.output_tokens))
    if os.path.isdir(args.output_counts) is False:
        raise ValueError("The directory for output of counts '%s' "
                         "does not exist" % (args.output_counts))

    # load metadata
    metadata = pd.read_csv("metadata/metadata.csv").set_index("id")

    # load languages dict
    langs_dict = get_langs_dict()

    # build job list
    jobs = []
    for filename in glob.glob(join(args.raw, 'PG%s_raw.txt' % (args.pattern))):
        if args.min_age_seconds > 0:
            mtime = os.path.getmtime(filename)
            if (time.time() - mtime) < args.min_age_seconds:
                if not args.quiet:
                    print("# WARNING: skipping recently modified '%s'" % filename)
                continue
        PG_id = filename.split("/")[-1].split("_")[0]
        try:
            language = "english"
            lang_id = ast.literal_eval(metadata.loc[PG_id, "language"])[0]
            if lang_id in langs_dict.keys():
                language = langs_dict[lang_id]
            pg_number = PG_id[2:]
            needed = not _outputs_exist(pg_number, args.output_text, args.output_tokens, args.output_counts)
            jobs.append((filename, language, needed))
        except KeyError:
            if not args.quiet:
                print("# WARNING: metadata for '%s' not found" % filename)
        except Exception:
            if not args.quiet:
                print("# WARNING: cannot process '%s' (unknown error during setup)" % filename)
                traceback.print_exc()

    # process jobs
    pbooks = 0
    new_books = 0
    skipped_existing = 0
    if args.workers <= 1:
        for filename, language, needed in jobs:
            fname, status, err, needed = _process_single(
                filename,
                language,
                args.output_text,
                args.output_tokens,
                args.output_counts,
                args.log_file,
                False,
                needed
            )
            if status is None:
                pbooks += 1
                if needed:
                    new_books += 1
                else:
                    skipped_existing += 1
                if not args.quiet:
                    print("Processed %d books (new: %d, skipped: %d)..." % (pbooks, new_books, skipped_existing), end="\r")
            elif status == "unicode":
                if not args.quiet:
                    print("# WARNING: cannot process '%s' (encoding not UTF-8)" % fname)
            else:
                if not args.quiet:
                    print("# WARNING: cannot process '%s' (unknown error)" % fname)
                    if err:
                        print(err, end="" if err.endswith("\n") else "\n")
    else:
        with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_file = {
                ex.submit(
                    _process_single,
                    filename,
                    language,
                    args.output_text,
                    args.output_tokens,
                    args.output_counts,
                    args.log_file,
                    True,
                    needed
                ): filename
                for filename, language, needed in jobs
            }
            for fut in cf.as_completed(fut_to_file):
                fname, status, err, needed = fut.result()
                if status is None:
                    pbooks += 1
                    if needed:
                        new_books += 1
                    else:
                        skipped_existing += 1
                    if not args.quiet:
                        print("Processed %d books (new: %d, skipped: %d)..." % (pbooks, new_books, skipped_existing), end="\r")
                elif status == "unicode":
                    if not args.quiet:
                        print("# WARNING: cannot process '%s' (encoding not UTF-8)" % fname)
                else:
                    if not args.quiet:
                        print("# WARNING: cannot process '%s' (unknown error)" % fname)
                        if err:
                            print(err, end="" if err.endswith("\n") else "\n")
