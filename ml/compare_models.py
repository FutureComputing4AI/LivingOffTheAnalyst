import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from optim_param import best_params
from ember_features import load_features


def accuracy(y_truth, y_pred):
    return np.sum(y_truth == y_pred) / len(y_truth)

def auc_low_fpr(y_truth, y_prob, max_fpr=0.01):
    return roc_auc_score(y_truth, y_prob, max_fpr=max_fpr)


def lgbm_all_base_grid():
    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    # convert yara matrix to dense
    # note: I timed converting the ember data to sparse, overall slower
    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()

    # baseline: ember only
    start = time.time()
    params = {"application": "binary"}
    lgbm_dataset = lgb.Dataset(X_train_ember, y_train)
    clf = lgb.train(params, lgbm_dataset)
    elapsed = time.time() - start

    y_pred = clf.predict(X_train_ember)
    y_pred = (y_pred > 0.5).astype(int)
    base_train_acc = accuracy(y_train, y_pred)

    y_pred = clf.predict(X_test_ember)
    y_pred = (y_pred > 0.5).astype(int)
    base_test_acc = accuracy(y_test, y_pred)
    
    y_prob = clf.predict(X_test_ember)
    base_test_auc = auc_low_fpr(y_test, y_prob)

    print("Baseline lgbm on ember: ", base_train_acc, base_test_acc, base_test_auc)
        
    with open('runs/base_lr_grid.pickle', 'rb') as f:
        lr_runs = pickle.load(f)

    run_info = []
    run = {'nnz': 0, 'acc': base_test_acc, 'auc': base_test_auc}
    run_info.append(run)

    for info in tqdm(lr_runs):
        lasso_coefs = info['coefs']
        support = np.where(lasso_coefs != 0)[0]
        
        if len(support) == 0:
            continue

        Xs_train_yara = X_train_yara[:, support]
        Xs_test_yara = X_test_yara[:, support]

        X_train = np.hstack([X_train_ember, Xs_train_yara])
        X_test = np.hstack([X_test_ember, Xs_test_yara])
        
        start = time.time()
        params = {"application": "binary"}
        lgbm_dataset = lgb.Dataset(X_train, y_train)
        clf = lgb.train(params, lgbm_dataset)
        elapsed = time.time() - start
        
        y_pred = clf.predict(X_train)
        y_pred = (y_pred > 0.5).astype(int)
        train_acc = accuracy(y_train, y_pred)

        y_pred = clf.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)
        test_acc = accuracy(y_test, y_pred)
                
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        
        y_prob = clf.predict(X_test)
        auc = auc_low_fpr(y_test, y_prob)

        run = {}
        run['nnz'] = info['nnz']
        run['train_acc'] = train_acc
        run['acc'] = test_acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

    with open('runs/lgb_all_base_grid.pickle', 'wb') as f:
        pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)


def rf_all_base_grid():
    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    # convert yara matrix to dense
    # note: I timed converting the ember data to sparse, overall slower
    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()

    # baseline: ember only
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train_ember, y_train)
    elapsed = time.time() - start

    y_pred = rf.predict(X_train_ember)
    base_train_acc = accuracy(y_train, y_pred)

    y_pred = rf.predict(X_test_ember)
    base_test_acc = accuracy(y_test, y_pred)
    
    y_prob = rf.predict_proba(X_test_ember)[:, 1]
    base_test_auc = auc_low_fpr(y_test, y_prob)

    print("Baseline rf on ember: ", base_train_acc, base_test_acc, base_test_auc)
        
    with open('runs/base_lr_grid.pickle', 'rb') as f:
        lr_runs = pickle.load(f)

    run_info = []
    run = {'nnz': 0, 'acc': base_test_acc, 'auc': base_test_auc}
    run_info.append(run)

    for info in tqdm(lr_runs):
        lasso_coefs = info['coefs']
        support = np.where(lasso_coefs != 0)[0]
        
        if len(support) == 0:
            continue

        Xs_train_yara = X_train_yara[:, support]
        Xs_test_yara = X_test_yara[:, support]

        X_train = np.hstack([X_train_ember, Xs_train_yara])
        X_test = np.hstack([X_test_ember, Xs_test_yara])
        
        start = time.time()
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = rf.predict(X_train)
        train_acc = accuracy(y_train, y_pred)

        y_pred = rf.predict(X_test)
        test_acc = accuracy(y_test, y_pred)
                
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        y_prob = rf.predict_proba(X_test)[:, 1]
        auc = auc_low_fpr(y_test, y_prob)

        run = {}
        run['nnz'] = info['nnz']
        run['train_acc'] = train_acc
        run['acc'] = test_acc
        run['auc_fpr_0.01'] = auc
        run['tn'] = tn
        run['fp'] = fp
        run['fn'] = fn
        run['tp'] = tp
        run['time'] = elapsed
        run_info.append(run)
        print(run)

    with open('runs/rf_all_base_grid.pickle', 'wb') as f:
        pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    
    lgbm_all_base_grid()
    
    rf_all_base_grid()
