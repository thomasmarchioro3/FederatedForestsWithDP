import tqdm
import argparse

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Local imports
from utils.decision_tree import DecisionTreeDP
from utils.federated_forest import FederatedForest


if __name__ == '__main__':

    argsparser = argparse.ArgumentParser()

    argsparser.add_argument("--epsilon", type=float, default=1)
    argsparser.add_argument("--max_depth", type=int, default=10)
    argsparser.add_argument("--n_trials", type=int, default=10)
    argsparser.add_argument("--random_seed", type=int, default=42)

    args = argsparser.parse_args()

    epsilon = args.epsilon
    max_depth = args.max_depth
    n_clients = args.n_clients
    n_trials = args.n_trials

    np.random.seed(args.random_seed)

    results_local = []
    results_global = []
    results_per_partition = []

    for trial in range(n_trials):

        epsilon_str = f'{epsilon:.2f}' if not isinstance(epsilon, int) else str(int(epsilon))

        print(f"Trial {trial+1}/{n_trials} - Epsilon: {epsilon_str}")

        df_test_global = pd.concat([pd.read_csv(f"metadata/test_{i+1:02d}.csv") for i in range(18)])
        X_test_global = df_test_global.drop('act', axis=1)
        y_test_global = df_test_global['act']
        label_encoder = LabelEncoder()
        y_test_global = label_encoder.fit_transform(y_test_global)

        forest = FederatedForest()

        for i in tqdm.tqdm(range(n_clients)):

            df_train_i = pd.read_csv(f"metadata/train_{i+1:02d}.csv")

            print(f"Training on train_{i+1:02d}.csv")

            X_train = df_train_i.drop('act', axis=1)
            y_train = df_train_i['act']

            y_train = label_encoder.transform(y_train)

            model = DecisionTreeDP(max_depth=max_depth, epsilon=epsilon)

            model.fit(X_train, y_train)

            forest.add_model(model)

            print(f"Testing on test_{i+1:02d}.csv")

            df_test_i = pd.read_csv(f"metadata/test_{i+1:02d}.csv")

            X_test = df_test_i.drop('act', axis=1)
            y_test = df_test_i['act']

            y_test = label_encoder.transform(y_test)

            y_pred = model.predict(X_test)

            acc_test_local = accuracy_score(y_test, y_pred)

            print(f"Local test accuracy: {acc_test_local}")

            results_local.append({'trial': trial+1, 'client': i, 'accuracy': acc_test_local})

            res_ = {10: [], 100: [], 1000: []}
            for j in range(n_clients):
                df_test_j = pd.read_csv(f"metadata/test_{j+1:02d}.csv")

                X_test_j = df_test_j.drop('act', axis=1)
                y_test_j = df_test_j['act']
                y_test_j = label_encoder.transform(y_test_j)

                y_pred_j = model.predict(X_test_j)
                

                for n_samples in res_.keys():
                    idx_j = np.random.choice(len(y_pred_j), n_samples, replace=False)
                    yy_true = y_test_j[idx_j]
                    yy_pred = y_pred_j[idx_j]
                    acc_test_j = accuracy_score(yy_true, yy_pred)

                    res_[n_samples].append(acc_test_j)
            
            res_per_partition = dict()
            res_per_partition['trial'] = trial + 1
            res_per_partition['client'] = i
            res_per_partition.update({n_samples: 1*(np.argmax(res_[n_samples])==i) for n_samples in res_.keys()})
            results_per_partition.append(res_per_partition)

        y_pred_global = forest.predict(X_test_global)

        acc_test_global = accuracy_score(y_test_global, y_pred_global)

        print(f"Global test accuracy: {acc_test_global}")

        results_global.append({'trial': trial + 1, 'accuracy': acc_test_global})

        df_local = pd.DataFrame(results_local)
        df_global = pd.DataFrame(results_global)
        df_per_partition = pd.DataFrame(results_per_partition)

        df_local.to_csv(f"results/federated/local_{epsilon_str}_d{max_depth}.csv", index=False)
        df_global.to_csv(f"results/federated/global_{epsilon_str}_d{max_depth}.csv", index=False)
        df_per_partition.to_csv(f"results/federated/per_partition_{epsilon_str}_d{max_depth}.csv", index=False)