import pickle
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split

from ember_features import load_features


best_params = {
    'booster': 'gbtree',
    'reg_lambda': 7e-05,
    'reg_alpha': 0.3,
    'subsample': 0.95,
    'colsample_bytree': 0.98,
    'max_depth': 9,
    'min_child_weight': 6,
    'eta': 0.4,
    'gamma': 0.2,
    'grow_policy': 'depthwise',
    "n_jobs": 32
}

def accuracy(y_truth, y_pred):
    return np.sum(y_truth == y_pred) / len(y_truth)


class Objective:
    def __init__(self, data):
        self.data = data

    def __call__(self, trial):
        X_train, y_train, X_test, y_test = self.data

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "tree_method": "auto",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            # L2 regularization weight.
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 1.0, log=True),
            # L1 regularization weight.
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        clf = xgb.XGBClassifier(**param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        val_acc = accuracy(y_test, y_pred)
        return val_acc


def test_final_param(X_train, y_train, X_test, y_test, params):
    # compute test set accuracy with params
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy(y_test, y_pred)
    print('Tuned test acc: ', test_acc)
    
    # compare with default
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy(y_test, y_pred)
    print('Default test acc: ', test_acc)


if __name__ == "__main__":
    
    train_data, test_data = load_features()
    X_train_ember, X_train_yara, y_train = train_data
    X_test_ember, X_test_yara, y_test = test_data

    X_train_yara = X_train_yara.toarray()
    X_test_yara = X_test_yara.toarray()

    # pick ~300 yara rules from independent grid
    with open('runs/base_lr_grid.pickle', 'rb') as f:
        lr_runs = pickle.load(f)
    for info in lr_runs:
        if info['nnz'] >= 300:
            break
    support = np.where(info['coefs'] != 0)[0]
    X_train_yara = X_train_yara[:, support]
    X_test_yara = X_test_yara[:, support]
    X_train = np.hstack([X_train_ember, X_train_yara])
    X_test = np.hstack([X_test_ember, X_test_yara])

    # do optimization over fixed validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0)
    
    objective = Objective((X_train, y_train, X_val, y_val))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=1)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    test_final_param(X_train, y_train, X_test, y_test, trial.params)
