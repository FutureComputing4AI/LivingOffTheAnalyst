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
        
    with open('runs/yara_analysis/lr_top_acc.pickle', 'rb') as f:
        acc_runs = pickle.load(f)

    with open('runs/yara_analysis/lr_top_precs.pickle', 'rb') as f:
        prec_runs = pickle.load(f)

    with open('runs/yara_analysis/lr_top_recalls.pickle', 'rb') as f:
        recall_runs = pickle.load(f)


    fig, ax = plt.subplots()

    # XGB block
    X, y = parse_nnz_vs_acc(acc_runs)
    ax.plot(X, y, '.-', label='Top accuracy')
    
    X, y = parse_nnz_vs_acc(prec_runs)
    ax.plot(X, y, '.-', label='Top precision')

    X, y = parse_nnz_vs_acc(recall_runs)
    ax.plot(X, y, '.-', label='Top recall')

    X, y = parse_nnz_vs_acc(lasso_runs)
    ax.plot(X, y, '.-', label='Lasso')
    

    ax.set_xlabel('YARA Rules Used')
    ax.set_ylabel('Test Accuracy')
    plt.legend()
    ax.set_xscale('log')
    plt.savefig('figs/feature_select.png', dpi=200)
    tikzplotlib.save('figs/feature_select.tex')

    plt.clf()
