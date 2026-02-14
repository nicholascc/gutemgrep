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
        src = "%s/%s" % (args.mirror_url, prefix) if prefix is not None else args.mirror_url
        sp_args = ["rsync", "-am%s" % vstring,
                   "--include", "*/",
                   "--include", "[p123456789][g0123456789]%s[.-][t0][x.]t[x.]*[t8]" % args.pattern,
                   "--exclude", "*",
                   src, args.mirror
                   ]
        return subprocess.call(sp_args)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip() != ""]
    if args.workers <= 1 or len(prefixes) <= 1:
        for p in prefixes:
            _rsync_prefix(p)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_rsync_prefix, prefixes))

    # Get rid of duplicates
    # ---------------------
    # A very small portion of books are stored more than
    # once in PG's site. We keep the newest one, see
    # erase_duplicates_in_mirror docstring.
    dups_list = list_duplicates_in_mirror(mirror_dir=args.mirror)

    # Populate raw from mirror
    # ------------------------
    # We populate 'raw_dir' hardlinking to
    # the hidden 'mirror_dir'. Names are standarized
    # into PG12345_raw.txt form.
    populate_raw_from_mirror(
        mirror_dir=args.mirror,
        raw_dir=args.raw,
        overwrite=args.overwrite_raw,
        dups_list=dups_list,
        quiet=args.quiet
        )

    # Update metadata
    # ---------------
    # By default, update the whole metadata csv
    # file each time new data is downloaded.
    make_df_metadata(
        path_xml=os.path.join(args.metadata, 'rdf-files.tar.bz2'),
        path_out=os.path.join(args.metadata, 'metadata.csv'),
        update=args.keep_rdf
        )

    # Bookshelves
    # -----------
    # Get bookshelves and their respective books and titles as dicts
    BS_dict, BS_num_to_category_str_dict = parse_bookshelves()
    with open("metadata/bookshelves_ebooks_dict.pkl", 'wb') as fp:
        pickle.dump(BS_dict, fp)
    with open("metadata/bookshelves_categories_dict.pkl", 'wb') as fp:
        pickle.dump(BS_num_to_category_str_dict, fp)
