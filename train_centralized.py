import argparse

import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import tqdm

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_trees", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    n_trials = args.trials
    n_trees = args.n_trees

    df_train_global = pd.concat([pd.read_csv(f"metadata/train_{i+1:02d}.csv") for i in range(18)])
    X_train_global = df_train_global.drop('act', axis=1)
    y_train_global = df_train_global['act']

    df_test_global = pd.concat([pd.read_csv(f"metadata/test_{i+1:02d}.csv") for i in range(18)])
    X_test_global = df_test_global.drop('act', axis=1)
    y_test_global = df_test_global['act']
    label_encoder = LabelEncoder()
    y_test_global = label_encoder.fit_transform(y_test_global)
    y_train_global = label_encoder.fit_transform(y_train_global)

    results_global = []

    for trial in tqdm.tqdm(range(n_trials)):

        random_state = random.randint(0, 100)

        model = RandomForestClassifier(n_estimators=n_trees, random_state=random_state)
        model.fit(X_train_global, y_train_global)
        y_pred = model.predict(X_test_global)
        acc_test = accuracy_score(y_test_global, y_pred)
        results_global.append(acc_test)

        print(f"Global test accuracy: {acc_test}")

    df_global = pd.DataFrame(results_global)
    df_global.to_csv("results/centralized/global.csv")