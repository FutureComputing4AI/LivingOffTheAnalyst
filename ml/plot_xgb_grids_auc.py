import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tikzplotlib


def parse_nnz_vs_auc(runs):
    nnzs = []
    accs = []
    for run in runs:
        if 'auc_fpr_0.01' in run.keys():
            nnzs.append(run['nnz'])
            accs.append(run['auc_fpr_0.01'])
        elif 'auc' in run.keys():
            nnzs.append(run['nnz'])
            accs.append(run['auc'])
    
    df = pd.DataFrame({'nnz': nnzs, 'auc': accs})
    df = df.sort_values('nnz')
    X = df['nnz'].values
    y = df['auc'].values
    X[X == 0] = 1       # for log-scale
    return X, y


if __name__ == '__main__':
        
    with open('runs/xgb_yara_base_grid.pickle', 'rb') as f:
        xgb_yara_runs = pickle.load(f)

    with open('runs/xgb_all_base_grid.pickle', 'rb') as f:
        xgb_all_base_runs = pickle.load(f)

    with open('runs/xgb_all_phat_grid.pickle', 'rb') as f:
        xgb_all_phat_runs = pickle.load(f)

    with open('runs/xgb_all_stack_grid.pickle', 'rb') as f:
        xgb_all_stack_runs = pickle.load(f)

    # obtained from print statement
    # ember_xgb_only = 0.91577      # default
    ember_xgb_only = 0.9357         # tuned for ember+yara
    
    fig, ax = plt.subplots()

    # XGB block
    X, y = parse_nnz_vs_auc(xgb_yara_runs)
    ax.plot(X, y, '.-', label='YARA only')
    
    X, y = parse_nnz_vs_auc(xgb_all_base_runs)
    ax.plot(X, y, '.-', label='EMBER, YARA independent')

    X, y = parse_nnz_vs_auc(xgb_all_phat_runs)
    ax.plot(X, y, '.-', label='EMBER, YARA conditional')

    X, y = parse_nnz_vs_auc(xgb_all_stack_runs)
    ax.plot(X, y, '.-', label='EMBER, YARA stacked')

    ax.axhline(y=ember_xgb_only, label='EMBER only', c='C6')

    ax.set_xlabel('YARA Rules Used')
    ax.set_ylabel('AUC at FPR < 0.01')
    plt.legend()
    ax.set_xscale('log')
    # ax.set_ylim([0.8, 1.0])
    ax.set_xlim([5, 5000])
    plt.savefig('figs/xgb_grids_auc.png', dpi=200)
    tikzplotlib.save('figs/xgb_grids_auc.tex')

    plt.clf()
