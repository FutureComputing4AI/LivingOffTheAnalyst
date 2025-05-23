import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from collections import defaultdict

def to_sparse(yara_infile: str, meta_infile: str, mat_outfile: str, yara_encoding: str="latin-1") -> None:
    # load ember's metadata
    metadata_df = pd.read_csv(meta_infile)
    metadata_df.set_index("sha256", inplace=True)

    # parse yara scan results
    yara_results = defaultdict(set)
    with open(yara_infile, 'r', encoding=yara_encoding) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2: continue

            rule, malware_hash = parts[:2]
            malware_hash = malware_hash.split('/')[-1].strip()
            yara_results[malware_hash].add(rule)

    # align the yara scan's results with the metadata
    aligned_yara_results = {k: v for k, v in yara_results.items() if k in metadata_df.index}

    # grab all rules with hits in the scan output
    rule_list = sorted(set(rule for rules in aligned_yara_results.values() for rule in rules))
    rule_to_idx = {rule: idx for idx, rule in enumerate(rule_list)}

    # prepare data for the sparse matrix
    rows, cols, data = [], [], []
    for sha256, rules in aligned_yara_results.items():
        row_idx = metadata_df.index.get_loc(sha256)

        for rule in rules:
            rows.append(row_idx)
            cols.append(rule_to_idx[rule])
            data.append(1)

    # create the sparse matrix
    matrix_shape = (len(metadata_df), len(rule_list))
    sparse_matrix = csr_matrix((data, (rows, cols)), shape=matrix_shape)

    # save the unsplit sparse matrix
    np.savez(
        mat_outfile,
        data=sparse_matrix.data,
        indices=sparse_matrix.indices,
        indptr=sparse_matrix.indptr,
        shape=sparse_matrix.shape,
        file_sha256=metadata_df.index.tolist(),
        rule_names=rule_list
    )

    return

# split the sparse matrix (generated above) into to train/test sets
def subset_split(mat_infile: str, meta_infile: str, npz_path: str) -> None:
    as_path = lambda fname: os.path.join(npz_path, fname)

    # load ember's metadata and the sparse matrix
    metadata_df = pd.read_csv(meta_infile)
    dataset = np.load(mat_infile)
    sparse = csr_matrix((dataset["data"], dataset["indices"], dataset["indptr"]), shape=dataset["shape"])

    # prepare train set.
    # NOTE: ember comes with some samples labeled with -1. we don't use these
    train_indices = metadata_df[(metadata_df["subset"] == "train") & (metadata_df["label"] != -1)].index
    X_train = sparse[train_indices]
    y_train = metadata_df.loc[train_indices, "label"]

    # prepare test set
    test_indices = metadata_df[(metadata_df["subset"] == "test") & (metadata_df["label"] != -1)].index
    X_test = sparse[test_indices]
    y_test = metadata_df.loc[test_indices, "label"]

    # dump the train set's data
    save_npz(as_path("train_matrix.npz"), X_train)
    np.savez(as_path("train_labels.npz"), labels=y_train)
    np.savez(as_path("train_hashes.npz"), sha256=metadata_df.loc[train_indices, "sha256"])

    # dump the test set's data
    save_npz(as_path("test_matrix.npz"), X_test)
    np.savez(as_path("test_labels.npz"), labels=y_test)
    np.savez(as_path("test_hashes.npz"), sha256=metadata_df.loc[test_indices, "sha256"])

    return
