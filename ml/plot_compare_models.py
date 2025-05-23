import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tikzplotlib


def parse_nnz_vs_acc(runs):
    nnzs = []
    accs = []
    for run in runs:
        nnzs.append(run['nnz'])
        accs.append(run['acc'])
    
    df = pd.DataFrame({'nnz': nnzs, 'acc': accs})
    df = df.sort_values('nnz')
    X = df['nnz'].values
    y = df['acc'].values
    X[X == 0] = 1       # for log-scale
    return X, y


if __name__ == '__main__':

    with open('runs/stack_lr_grid.pickle', 'rb') as f:
        lr_all_runs = pickle.load(f)
        
    with open('runs/xgb_all_base_grid.pickle', 'rb') as f:
        xgb_all_base_runs = pickle.load(f)

    with open('runs/lgb_all_base_grid.pickle', 'rb') as f:
        lgb_all_base_runs = pickle.load(f)

    with open('runs/rf_all_base_grid.pickle', 'rb') as f:
        rf_all_base_runs = pickle.load(f)
        
    # obtained from print statement
    ember_xgb_only = 0.96557         # tuned for ember+yara

    fig, ax = plt.subplots()
    
    X, y = parse_nnz_vs_acc(lr_all_runs)
    ax.plot(X, y, '.-', label='Logistic Reg.')

    X, y = parse_nnz_vs_acc(xgb_all_base_runs)
    ax.plot(X, y, '.-', label='XGBoost')

    X, y = parse_nnz_vs_acc(lgb_all_base_runs)
    ax.plot(X, y, '.-', label='LightGBM')

    X, y = parse_nnz_vs_acc(rf_all_base_runs)
    ax.plot(X, y, '.-', label='Random Forest')
    
    # ax.axhline(y=ember_xgb_only, label='EMBER only', c='C6')

    ax.set_xlabel('YARA Rules Used')
    ax.set_ylabel('Test Accuracy')
    plt.legend()
    ax.set_xscale('log')
    ax.set_ylim([0.6, 1.0])
    ax.set_xlim([10, 4000])
    plt.savefig('figs/compare_models.png', dpi=200)
    tikzplotlib.save('figs/compare_models.tex')

    plt.clf()
