import pandas as pd

# from decisiontree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.decision_tree import DecisionTreeDP

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    df_test_global = pd.concat([pd.read_csv(f"metadata/test_{i+1:02d}.csv") for i in range(18)])
    X_test_global = df_test_global.drop('act', axis=1)
    y_test_global = df_test_global['act']
    label_encoder = LabelEncoder()
    y_test_global = label_encoder.fit_transform(y_test_global)

    results_local = []
    results_global = []
    results_per_partition = []

    for i in range(18):

        df_train_i = pd.read_csv(f"metadata/train_{i+1:02d}.csv")
        df_test_i = pd.read_csv(f"metadata/test_{i+1:02d}.csv")

        print(f"Training on train_{i+1:02d}.csv")

        X_train = df_train_i.drop('act', axis=1)
        y_train = df_train_i['act']
        y_train = label_encoder.transform(y_train)
        X_test = df_test_i.drop('act', axis=1)
        y_test = df_test_i['act']
        y_test = label_encoder.transform(y_test)

        model = DecisionTreeClassifier()
        # model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = DecisionTreeDP(max_depth=5, epsilon=100)
        model.fit(X_train, y_train)

        print(f"Testing on test_{i+1:02d}.csv")

        y_pred = model.predict(X_test)

        acc_test_local = accuracy_score(y_test, y_pred)

        print(f"Local test accuracy: {acc_test_local}")

        results_local.append(acc_test_local)

        res_ = []

        for j in range(18):
            df_test_j = pd.read_csv(f"metadata/test_{j+1:02d}.csv")

            X_test_j = df_test_j.drop('act', axis=1)
            y_test_j = df_test_j['act']
            y_test_j = label_encoder.transform(y_test_j)

            y_pred_j = model.predict(X_test_j)
            acc_test_j = accuracy_score(y_test_j, y_pred_j)

            res_.append(acc_test_j)

        results_per_partition.append(res_)

        y_pred_global = model.predict(X_test_global)
        acc_test_global = accuracy_score(y_test_global, y_pred_global)

        print(f"Global test accuracy: {acc_test_global}")

        results_global.append(acc_test_global)
    
    df_local = pd.DataFrame(results_local)
    df_global = pd.DataFrame(results_global)
    df_per_partition = pd.DataFrame(results_per_partition)

    df_local.to_csv("results/standalone/local.csv")
    df_global.to_csv("results/standalone/global.csv")
    df_per_partition.to_csv("results/standalone/per_partition.csv")