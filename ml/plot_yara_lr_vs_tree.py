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
        
    with open('runs/base_lr_grid.pickle', 'rb') as f:
        lasso_runs = pickle.load(f)
        
    with open('runs/xgb_yara_base_grid.pickle', 'rb') as f:
        xgb_runs = pickle.load(f)


    fig, ax = plt.subplots()

    # XGB block
    X, y = parse_nnz_vs_acc(lasso_runs)
    ax.plot(X, y, '.-', label='Linear Model')
    
    X, y = parse_nnz_vs_acc(xgb_runs)
    ax.plot(X, y, '.-', label='XGBoost')


    ax.set_xlabel('YARA Rules Used')
    ax.set_ylabel('Test Accuracy')
    plt.legend()
    ax.set_xscale('log')
    plt.savefig('figs/yara_lr_vs_tree.png', dpi=200)
    tikzplotlib.save('figs/yara_lr_vs_tree.tex')

    plt.clf()
