''' Loading function for our datasets '''
import os
import time
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix, csr_matrix, hstack
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

EMBER_DIR = ''

"""
class PEFeatureExtractor(object):
    def __init__(self, _feature_version=2, print_feature_warning=True, features_file=''):
        self.features = []
        features = {
                    'ByteHistogram': ByteHistogram(),  # 256
                    'ByteEntropyHistogram': ByteEntropyHistogram(),  # 256
                    'StringExtractor': StringExtractor(),  # 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1
                    'GeneralFileInfo': GeneralFileInfo(),  # 10
                    'HeaderFileInfo': HeaderFileInfo(),  # 62
                    'SectionInfo': SectionInfo(),  # 5 + 50 + 50 + 50 + 50 + 50
                    'ImportsInfo': ImportsInfo(),  # 1280
                    'ExportsInfo': ExportsInfo()  # 128
            }
        # ... snip ...
"""

#def initial_setup():    
#    ember.create_vectorized_features(EMBER_DIR)
#    ember.create_metadata(EMBER_DIR)

def read_vectorized_features(data_dir, subset=None, feature_version=2):
    if subset is not None and subset not in ["train", "test"]:
        return None

    #ndim = PEFeatureExtractor(feature_version).dim
    ndim = sum([
      256,
      256,
      1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1,
      10,
      62,
      5 + 50 + 50 + 50 + 50 + 50,
      1280,
      128
    ])
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
        N = y_train.shape[0]
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
        N = y_test.shape[0]
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test

def load_features():
    ''' This function ensures that the standard EMBER features and YARA features
    are split across the same train and test sets, and are in the correct order.

    As a result the EMBER and YARA features can be horizontally stacked, since
    any row of both matrices refer to the same sample.
    '''
    X_train_ember, y_train_ember, X_test_ember, y_test_ember = read_vectorized_features(EMBER_DIR)

    # (confirmed) the entries of data matrices are ordered w.r.t. the metadata
    train_metadata = pd.read_csv(os.path.join(EMBER_DIR, 'train_metadata.csv'))
    test_metadata = pd.read_csv(os.path.join(EMBER_DIR, 'test_metadata.csv'))

    # load the hashes for the processed yara train/test splits
    yara_train_hash = np.load('data/train_hashes.npz', allow_pickle=True)
    yara_train_hash = pd.DataFrame({'yara_hash': yara_train_hash['sha256']})

    yara_test_hash = np.load('data/test_hashes.npz', allow_pickle=True)
    yara_test_hash = pd.DataFrame({'yara_hash': yara_test_hash['sha256']})

    # merge yara and ember2018 hashes to get yara train and test splits
    train_info = train_metadata[['Unnamed: 0', 'sha256']].merge(
            yara_train_hash, left_on='sha256', right_on='yara_hash')
    train_idx = train_info['Unnamed: 0'].values

    test_info = test_metadata[['Unnamed: 0', 'sha256']].merge(
            yara_test_hash, left_on='sha256', right_on='yara_hash')
    test_idx = test_info['Unnamed: 0'].values

    X_train_ember = X_train_ember[train_idx, :]
    X_test_ember = X_test_ember[test_idx, :]

    # load corresponding yara datasets
    #X_train_yara = csc_matrix(load_npz('data/train_matrix.npz'))
    X_train_yara = None
    y_train = np.load('data/train_labels.npz')['labels']

    #X_test_yara = csc_matrix(load_npz('data/test_matrix.npz'))
    X_test_yara = None
    y_test = np.load('data/test_labels.npz')['labels']

    return (X_train_ember, X_train_yara, y_train), (X_test_ember, X_test_yara, y_test)

if __name__ == '__main__':
    #initial_setup()
    load_features()
