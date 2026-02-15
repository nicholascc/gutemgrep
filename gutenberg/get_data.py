"""
Project Gutenberg parsing with python 3.

Written by
M. Gerlach & F. Font-Clos

"""
from src.utils import populate_raw_from_mirror, list_duplicates_in_mirror
from src.metadataparser import make_df_metadata
from src.bookshelves import get_bookshelves
from src.bookshelves import parse_bookshelves

import argparse
import os
import subprocess
import pickle
import concurrent.futures as cf
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Update local PG repository.\n\n"
        "This script will download all books currently not in your\n"
        "local copy of PG and get the latest version of the metadata.\n"
        )
    # mirror dir
    parser.add_argument(
        "-m", "--mirror",
        help="Path to the mirror folder that will be updated via rsync.",
        default='data/.mirror/',
        type=str)

    # raw dir
    parser.add_argument(
        "-r", "--raw",
        help="Path to the raw folder.",
        default='data/raw/',
        type=str)

    # metadata dir
    parser.add_argument(
        "-M", "--metadata",
        help="Path to the metadata folder.",
        default='metadata/',
        type=str)

    # pattern matching
    parser.add_argument(
        "-p", "--pattern",
        help="Patterns to get only a subset of books.",
        default='*',
        type=str)

    # update argument
    parser.add_argument(
        "-k", "--keep_rdf",
        action="store_false",
        help="If there is an RDF file in metadata dir, do not overwrite it.")

    # update argument
    parser.add_argument(
        "-owr", "--overwrite_raw",
        action="store_true",
        help="Overwrite files in raw.")

    # mirror url
    parser.add_argument(
        "--mirror_url",
        help="Rsync mirror URL (module), e.g. 'gutenberg.pglaf.org::gutenberg'",
        default="gutenberg.pglaf.org::gutenberg",
        type=str)

    # parallel rsync workers (prefix-sharded by top-level digit dirs)
    parser.add_argument(
        "-w", "--workers",
        help="Number of parallel rsync workers (default: 1)",
        default=1,
        type=int)

    parser.add_argument(
        "--prefixes",
        help="Comma-separated top-level directory prefixes to rsync (default: 0-9)",
        default="0,1,2,3,4,5,6,7,8,9",
        type=str)

    parser.add_argument(
        "--prefix_depth",
        help="Generate prefixes of depth N (e.g. 2 -> 00..99). Overrides --prefixes when >1.",
        default=1,
        type=int)

    parser.add_argument(
        "--probe_prefixes",
        help="Probe mirror to discover directory prefixes for --prefix_depth > 1.",
        action="store_true")

    parser.add_argument(
        "--loop",
        help="Repeat rsync + populate_raw in a loop (useful for periodic saves).",
        action="store_true")

    parser.add_argument(
        "--sleep_seconds",
        help="Sleep time between loop iterations (default: 60).",
        default=60,
        type=int)

    parser.add_argument(
        "--update_metadata_each_pass",
        help="Update metadata/bookshelves on every loop pass (default: only after loop ends).",
        action="store_true")

    # quiet argument, to supress info
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode, do not print info, warnings, etc"
        )

    # create the parser
    args = parser.parse_args()

    # check that all dirs exist
    if not os.path.isdir(args.mirror):
        raise ValueError("The specified mirror directory does not exist.")
    if not os.path.isdir(args.raw):
        raise ValueError("The specified raw directory does not exist.")
    if not os.path.isdir(args.metadata):
        raise ValueError("The specified metadata directory does not exist.")

    # Update the .mirror directory via rsync
    # --------------------------------------
    # We sync the 'mirror_dir' with PG's site via rsync
    # The matching pattern, explained below, should match
    # only UTF-8 files.

    # pass the -v flag to rsync if not in quiet mode
    if args.quiet:
        vstring = ""
    else:
        vstring = "v"

    # Pattern to match the +  but not the - :
    #
    # + 12345 .   t   x  t .            utf  8
    # - 12345 .   t   x  t .      utf8 .gzi  p
    # + 12345 -   0   .  t x                 t 
    #---------------------------------------------
    #        [.-][t0][x.]t[x.]    *         [t8]
    def _rsync_prefix(prefix):
        src = "%s/%s" % (args.mirror_url.rstrip("/"), prefix) if prefix is not None else args.mirror_url
        sp_args = ["rsync", "-am%s" % vstring,
                   "--include", "*/",
                   "--include", "[p123456789][g0123456789]%s[.-][t0][x.]t[x.]*[t8]" % args.pattern,
                   "--exclude", "*",
                   src, args.mirror
                   ]
        return subprocess.call(sp_args)

    def _list_subdirs(prefix):
        src = "%s/%s/" % (args.mirror_url.rstrip("/"), prefix) if prefix else "%s/" % args.mirror_url.rstrip("/")
        try:
            res = subprocess.check_output(["rsync", "--list-only", src], stderr=subprocess.DEVNULL)
            lines = res.decode("utf-8", errors="ignore").splitlines()
            subdirs = []
            for line in lines:
                if not line:
                    continue
                if line[0] == "d":
                    name = line.split()[-1]
                    if name not in (".", ".."):
                        subdirs.append(name)
            return subdirs
        except Exception:
            return []

    if args.prefix_depth > 1:
        digits = ["%d" % i for i in range(10)]
        if args.probe_prefixes:
            prefixes = _list_subdirs("")
            if not prefixes:
                prefixes = digits
            for _ in range(args.prefix_depth - 1):
                next_prefixes = []
                for p in prefixes:
                    subdirs = _list_subdirs(p)
                    if subdirs:
                        next_prefixes.extend(["%s/%s" % (p, s) for s in subdirs])
                prefixes = next_prefixes or prefixes
        else:
            prefixes = digits
            for _ in range(args.prefix_depth - 1):
                prefixes = ["%s/%s" % (a, b) for a in prefixes for b in digits]
    else:
        prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip() != ""]
    if args.workers <= 1 or len(prefixes) <= 1:
        for p in prefixes:
            _rsync_prefix(p)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_rsync_prefix, prefixes))

    def _run_rsync():
        if args.workers <= 1 or len(prefixes) <= 1:
            for p in prefixes:
                _rsync_prefix(p)
        else:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(_rsync_prefix, prefixes))

    def _populate_raw():
        dups_list = list_duplicates_in_mirror(mirror_dir=args.mirror)
        populate_raw_from_mirror(
            mirror_dir=args.mirror,
            raw_dir=args.raw,
            overwrite=args.overwrite_raw,
            dups_list=dups_list,
            quiet=args.quiet
            )

    def _update_metadata_and_bookshelves():
        make_df_metadata(
            path_xml=os.path.join(args.metadata, 'rdf-files.tar.bz2'),
            path_out=os.path.join(args.metadata, 'metadata.csv'),
            update=args.keep_rdf
            )
        BS_dict, BS_num_to_category_str_dict = parse_bookshelves()
        with open("metadata/bookshelves_ebooks_dict.pkl", 'wb') as fp:
            pickle.dump(BS_dict, fp)
        with open("metadata/bookshelves_categories_dict.pkl", 'wb') as fp:
            pickle.dump(BS_num_to_category_str_dict, fp)

    try:
        while True:
            _run_rsync()
            _populate_raw()
            if args.update_metadata_each_pass:
                _update_metadata_and_bookshelves()
            if not args.loop:
                break
            time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        pass

    if not args.update_metadata_each_pass:
        _update_metadata_and_bookshelves()
    with open("metadata/bookshelves_ebooks_dict.pkl", 'wb') as fp:
        pickle.dump(BS_dict, fp)
    with open("metadata/bookshelves_categories_dict.pkl", 'wb') as fp:
        pickle.dump(BS_num_to_category_str_dict, fp)
