import os
import ember
import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse import csc_matrix, csr_matrix, hstack
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import tikzplotlib

from ember_features import load_features


def accuracy(y_truth, y_pred):
    return np.sum(y_truth == y_pred) / len(y_truth)

def precision(y_truth, y_pred):
    if np.sum(y_pred == 1) == 0:
        return 0
    return np.sum((y_pred == 1) & (y_truth == 1)) / np.sum(y_pred == 1)
    
def recall(y_truth, y_pred):
    return np.sum((y_pred == 1) & (y_truth == 1)) / np.sum(y_truth == 1)

def auc_low_fpr(y_truth, y_prob, max_fpr=0.01):
    return roc_auc_score(y_truth, y_prob, max_fpr=max_fpr)

# def recall_at_fpr(y_truth, y_prob):
#     thresholds = np.sort(y_prob)[::-1]
#     for thr in thresholds:
#         pred = (y_prob >= thr).astype(int)


def train_lr_over_grid(grid_filter, grid_name):
    run_info = []
    for nr in tqdm([1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500]):
        X_train = X_train_yara[:, grid_filter[:nr]]
        X_test = X_test_yara[:, grid_filter[:nr]]
        
        start = time.time()
        lr = LogisticRegression(C=10, penalty='l2', max_iter=100)
        lr.fit(X_train, y_train)
        elapsed = time.time() - start
        
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        acc = accuracy(y_test, y_pred)
        auc = auc_low_fpr(y_test, y_prob)
        
        run = {}
        run['nnz'] = nr
        run['coefs'] = lr.coef_[0]
        run['acc'] = acc
        run['auc_fpr_0.01'] = auc
        run['time'] = elapsed
        run_info.append(run)
        print(run)
        
        with open(f'runs/yara_analysis/{grid_name}.pickle', 'wb') as f:
            pickle.dump(run_info, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_lr_feat_selection(idx_acc_desc, idx_precs_desc, idx_recalls_desc):
    train_lr_over_grid(idx_acc_desc, 'lr_top_acc')
    train_lr_over_grid(idx_precs_desc, 'lr_top_precs')
    train_lr_over_grid(idx_recalls_desc, 'lr_top_recalls')
    return


if __name__ == '__main__':

    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()
    print(X_train_yara.shape)

    accs = [accuracy(y_train, X_train_yara[:, i]) for i in range(X_train_yara.shape[1])]
    precs = [precision(y_train, X_train_yara[:, i]) for i in range(X_train_yara.shape[1])]
    recalls = [recall(y_train, X_train_yara[:, i]) for i in range(X_train_yara.shape[1])]
    
    idx_acc_desc = np.argsort(accs)[::-1]
    idx_precs_desc = np.argsort(precs)[::-1]
    idx_recalls_desc = np.argsort(recalls)[::-1]
    
    run_lr_feat_selection(idx_acc_desc, idx_precs_desc, idx_recalls_desc)

    plt.clf()
    plt.violinplot([accs, precs, recalls],
                widths=0.5,
                showmedians=True,
                showextrema=True,
                points=1000,
                bw_method='scott',
                )
    ax = plt.gca()
    ax.set_xticks(np.arange(1, 4), labels=['Accuracy', 'Precision', 'Recall'])
    ax.set_xlim(0.25, 3 + 0.75)
    tikzplotlib.save('figs/metric_dist.tex')


    plt.clf()
    top_acc_precs = [precs[i] for i in idx_acc_desc[:200]]
    top_acc_recs = [recalls[i] for i in idx_acc_desc[:200]]
    plt.scatter(top_acc_precs, top_acc_recs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('figs/pvr_of_top_accs.png')
    tikzplotlib.save('figs/pvr_of_top_accs.tex')
    print(accs[idx_acc_desc[0]], accs[idx_acc_desc[199]])


    plt.clf()
    idx = np.where(y_train == 1)[0]
    X_mal = X_train_yara[idx]

    idx = np.where(y_train == 0)[0]
    X_ben = X_train_yara[idx]

    mal_hits = np.sum(X_mal, axis=0)
    ben_hits = np.sum(X_ben, axis=0)
    plt.plot(np.sort(mal_hits), np.linspace(0, 1, len(mal_hits), endpoint=False), label='Malware')
    plt.plot(np.sort(ben_hits), np.linspace(0, 1, len(ben_hits), endpoint=False), label='Benign')

    plt.ylim([0.55, 1])
    plt.xlim([10, 100000])
    plt.xscale('log')
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(5, 3)

    plt.show()