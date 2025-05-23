import os
#import ember
import time
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix, csr_matrix, hstack

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_auc_score
import xgboost as xgb

from optim_param import best_params
from ember_features import load_features


def accuracy(y_truth, y_pred):
    return np.sum(y_truth == y_pred) / len(y_truth)

def auc_low_fpr(y_truth, y_prob, max_fpr=0.01):
    return roc_auc_score(y_truth, y_prob, max_fpr=max_fpr)

def basic_lasso_grid():
    ''' Generates mapping of lambda -> selected features for IS '''
    # convert CSR --> CSC. is noticeably faster for liblinear interface
    X_train = csc_matrix(load_npz('data/train_matrix.npz'))
    y_train = np.load('data/train_labels.npz')['labels']

    X_test = csc_matrix(load_npz('data/test_matrix.npz'))
    y_test = np.load('data/test_labels.npz')['labels']

    C_min_exp = -5
    C_max_exp = 1

    run_info = []

    for C in tqdm(np.logspace(C_min_exp, C_max_exp, 20)):
        start = time.time()
        lr = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=100)
        lr.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        nnz = np.sum(lr.coef_[0] != 0)
        acc = accuracy(y_test, y_pred)
        auc = auc_low_fpr(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        run = {}
        run['C'] = C
        run['coefs'] = lr.coef_[0]
        run['nnz'] = nnz
        run['acc'] = acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

        with open('runs/base_lr_grid.pickle', 'wb') as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        if nnz > 4000:
            return

def conditional_lasso_grid():
    ''' Generates mapping of lambda -> selected features for CS '''
    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    # convert yara matrix to dense
    # note: I timed converting the ember data to sparse, overall slower
    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()

    # fit ember-only baseline to get probability vector
    print("Loaded ember features.")
    start = time.time()
    clf = xgb.XGBClassifier(**best_params)
    y_prob_train = cross_val_predict(    # train set needs to be done via kfold
        clf, X_train_ember, y_train, cv=5, method='predict_proba')
    y_prob_train = y_prob_train[:, [1]]

    # refit on whole data for test predictions
    clf = xgb.XGBClassifier(**best_params)
    clf.fit(X_train_ember, y_train)
    y_prob_test = clf.predict_proba(X_test_ember)
    y_prob_test = y_prob_test[:, [1]]
    elapsed = time.time() - start

    base_train_acc = accuracy(y_train, (y_prob_train.flatten() > 0.5).astype(int))
    base_test_acc = accuracy(y_test, (y_prob_test.flatten() > 0.5).astype(int))
    print("Baseline xgb on ember: ", base_train_acc, base_test_acc)

    X_train = np.hstack([y_prob_train, X_train_yara])
    X_test = np.hstack([y_prob_test, X_test_yara])

    C_min_exp = -5
    C_max_exp = 1

    run_info = []

    for C in tqdm(np.logspace(C_min_exp, C_max_exp, 20)):
        start = time.time()
        lr = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=50)
        lr.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        acc = accuracy(y_test, y_pred)
        auc = auc_low_fpr(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        # first coef is ember p_hat, relabel accordingly
        yara_coefs = lr.coef_[0][1:]
        nnz = np.sum(yara_coefs != 0)

        run = {}
        run['C'] = C
        run['coefs'] = yara_coefs
        run['nnz'] = nnz
        run['acc'] = acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

        with open('runs/phat_lr_grid.pickle', 'wb') as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        if nnz > 4000:
            return


def stacked_lasso_grid():
    ''' Generates mapping of lambda -> selected features for SFS '''
    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    # convert yara matrix to dense
    # note: I timed converting the ember data to sparse, overall slower
    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()

    X_train = np.hstack([X_train_ember, X_train_yara])
    X_test = np.hstack([X_test_ember, X_test_yara])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_ember_ft = X_train_ember.shape[1]
    print('ember feats: ', n_ember_ft)

    C_min_exp = -5
    C_max_exp = 1

    run_info = []

    for C in tqdm(np.logspace(C_min_exp, C_max_exp, 20)):
        start = time.time()
        lr = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=50)
        lr.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        acc = accuracy(y_test, y_pred)
        auc = auc_low_fpr(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        # first group of coefs are ember, relabel accordingly
        yara_coefs = lr.coef_[0][n_ember_ft:]
        nnz = np.sum(yara_coefs != 0)

        run = {}
        run['C'] = C
        run['coefs'] = yara_coefs
        run['nnz'] = nnz
        run['acc'] = acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

        with open('runs/stack_lr_grid.pickle', 'wb') as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        if nnz > 4000:
            return


def ember_lasso_grid():
    ''' this is a lasso grid over only the ember features (rather than using 
    all of them from the get-go). Not used in the paper. '''
    # load and keep ember features
    train_data, test_data = load_features()
    X_train, _, y_train = train_data
    X_test, _, y_test = test_data

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    C_min_exp = -5
    C_max_exp = 1

    run_info = []

    for C in tqdm(np.logspace(C_min_exp, C_max_exp, 20)):
        start = time.time()
        lr = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=100)
        lr.fit(X_train, y_train)
        elapsed = time.time() - start
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        nnz = np.sum(lr.coef_[0] != 0)
        acc = accuracy(y_test, y_pred)
        auc = auc_low_fpr(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        run = {}
        run['C'] = C
        run['coefs'] = lr.coef_[0]
        run['nnz'] = nnz
        run['acc'] = acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

        with open('runs/ember_lr_grid.pickle', 'wb') as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        if nnz > 2000:
            return


if __name__ == '__main__':
    basic_lasso_grid()

    # uncomment when load_features() is properly fixed
    #conditional_lasso_grid()
    #stacked_lasso_grid()
    #ember_lasso_grid()
