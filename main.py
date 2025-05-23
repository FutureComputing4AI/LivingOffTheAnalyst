import argparse
import os
import lib

JSON_PATH: str = "data/json"
CSV_PATH: str = "data/csv"
TXT_PATH: str = "data/txt"
NPZ_PATH: str = "data/npz"
YAR_PATH: str = "data/yar"

# step 1: extract info from yara existing yara rules
def extract_yara_rules(yara_rules_tld: str) -> None:
    print("Enumerating Yara rules...")
    lib.strings_as_yara_rules(
        yara_rules_tld,
        os.path.join(YAR_PATH, "rules.yar")
    )

    return

# step 2: run them over a dataset (do that yourself)

# step 3: convert yara output into matrices
def gen_split_matrices(yara_outfile: str) -> None:
    # step 3.1: into one 'master' matrix
    print("Constructing sparse matrix...")
    #os.path.join(CSV_PATH, "yara-out.csv"),
    lib.to_sparse(
        yara_outfile,
        os.path.join(CSV_PATH, "metadata.csv"),
        os.path.join(NPZ_PATH, "all_matrix.npz")
    )

    # step 3.2: into sparse matrices by train/test split
    print("Splitting sparse matrix...")
    lib.subset_split(
        os.path.join(NPZ_PATH, "all_matrix.npz"),
        os.path.join(CSV_PATH, "metadata.csv"),
        NPZ_PATH
    )

    return

# step 5: use the matrices

def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["extract", "matrix"],
        type=str,
        help="Pass 'extract' if Yara rules should be re-extracted.\nPass 'matrix' if features should be thrown into a matrix."
    )

    parser.add_argument(
        "--yara-rules-tld",
        type=str,
        help="Required if 'action' is 'extract.'\nShould point to the directory of Yara rules to harvest features from."
    )

    parser.add_argument(
        "--yara-outfile",
        type=str,
        help="Required if 'action' is 'matrix.'\nShould point to the Yara scan's stdout file."
    )

    args: argparse.Namespace = parser.parse_args()
    if args.action == "extract" and args.yara_rules_tld is None:
        parser.error("'--yara-rules-tld' is required if 'action' is 'extract.'")
    if args.action == "matrix" and args.yara_outfile is None:
        parser.error("'--yara-outfile' is required if 'action' is 'matrix.'")

    return args

def main() -> None:
    args: argparse.Namespace = parse_args()

    # extract and generate a Yara ruleset
    if args.action == "extract":
        extract_yara_rules(args.yara_rules_tld)

    # convert a Yara scan output into sparse matrices
    else:
        gen_split_matrices(args.yara_outfile)

    return

if __name__ == "__main__":
    main()
