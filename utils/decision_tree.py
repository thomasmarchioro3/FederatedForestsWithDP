import pandas as pd
import numpy as np

ACC_SCALE = 5_000
GYRO_SCALE = 5_000
EMG_SCALE = 100

DEFAULT_FEATURE_MINS = {
    'acc_rf_x': -ACC_SCALE,
    'acc_rf_y': -ACC_SCALE,
    'acc_rf_z': -ACC_SCALE,
    'gyro_rf_x': -GYRO_SCALE,
    'gyro_rf_y': -GYRO_SCALE,
    'gyro_rf_z': -GYRO_SCALE,
    'acc_rs_x': -ACC_SCALE,
    'acc_rs_y': -ACC_SCALE,
    'acc_rs_z': -ACC_SCALE,
    'gyro_rs_x': -GYRO_SCALE,
    'gyro_rs_y': -GYRO_SCALE,
    'gyro_rs_z': -GYRO_SCALE,
    'acc_rt_x': -ACC_SCALE,
    'acc_rt_y': -ACC_SCALE,
    'acc_rt_z': -ACC_SCALE,
    'gyro_rt_x': -GYRO_SCALE,
    'gyro_rt_y': -GYRO_SCALE,
    'gyro_rt_z': -GYRO_SCALE,
    'acc_lf_x': -ACC_SCALE,
    'acc_lf_y': -ACC_SCALE,
    'acc_lf_z': -ACC_SCALE,
    'gyro_lf_x': -GYRO_SCALE,
    'gyro_lf_y': -GYRO_SCALE,
    'gyro_lf_z': -GYRO_SCALE,
    'acc_ls_x': -ACC_SCALE,
    'acc_ls_y': -ACC_SCALE,
    'acc_ls_z': -ACC_SCALE,
    'gyro_ls_x': -GYRO_SCALE,
    'gyro_ls_y': -GYRO_SCALE,
    'gyro_ls_z': -GYRO_SCALE,
    'acc_lt_x': -ACC_SCALE,
    'acc_lt_y': -ACC_SCALE,
    'acc_lt_z': -ACC_SCALE,
    'gyro_lt_x': -GYRO_SCALE,
    'gyro_lt_y': -GYRO_SCALE,
    'gyro_lt_z': -GYRO_SCALE,
    'EMG_r': 0,
    'EMG_l': 0,
}

DEFAULT_FEATURE_MAXS = {
    'acc_rf_x': ACC_SCALE,
    'acc_rf_y': ACC_SCALE,
    'acc_rf_z': ACC_SCALE,
    'gyro_rf_x': ACC_SCALE,
    'gyro_rf_y': ACC_SCALE,
    'gyro_rf_z': ACC_SCALE,
    'acc_rs_x': ACC_SCALE,
    'acc_rs_y': ACC_SCALE,
    'acc_rs_z': ACC_SCALE,
    'gyro_rs_x': ACC_SCALE,
    'gyro_rs_y': ACC_SCALE,
    'gyro_rs_z': ACC_SCALE,
    'acc_rt_x': ACC_SCALE,
    'acc_rt_y': ACC_SCALE,
    'acc_rt_z': ACC_SCALE,
    'gyro_rt_x': ACC_SCALE,
    'gyro_rt_y': ACC_SCALE,
    'gyro_rt_z': ACC_SCALE,
    'acc_lf_x': ACC_SCALE,
    'acc_lf_y': ACC_SCALE,
    'acc_lf_z': ACC_SCALE,
    'gyro_lf_x': ACC_SCALE,
    'gyro_lf_y': ACC_SCALE,
    'gyro_lf_z': ACC_SCALE,
    'acc_ls_x': ACC_SCALE,
    'acc_ls_y': ACC_SCALE,
    'acc_ls_z': ACC_SCALE,
    'gyro_ls_x': ACC_SCALE,
    'gyro_ls_y': ACC_SCALE,
    'gyro_ls_z': ACC_SCALE,
    'acc_lt_x': ACC_SCALE,
    'acc_lt_y': ACC_SCALE,
    'acc_lt_z': ACC_SCALE,
    'gyro_lt_x': ACC_SCALE,
    'gyro_lt_y': ACC_SCALE,
    'gyro_lt_z': ACC_SCALE,
    'EMG_r': EMG_SCALE,
    'EMG_l': EMG_SCALE,
}

def true_divide(a, b):
    if b == 0:
        if a == 0:
            return 0
        else:
            return np.inf
    return a / b

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeDP:
    def __init__(self, max_depth=None, epsilon=None, feature_mins=list(DEFAULT_FEATURE_MINS.values()), feature_maxs=list(DEFAULT_FEATURE_MAXS.values()), random_state=None):
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.tree_ = None

        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs

        self.random_state = random_state

        self.total_queries = 0
        self.max_queries = (2**self.max_depth)*(len(self.feature_mins))
        if self.epsilon is not None:
            assert self.max_depth is not None
            self.noise_scale = self.max_queries / self.epsilon

    def _gini(self, counts):
        tot_counts = sum(counts)
        return 1.0 - sum(true_divide(counts[c], tot_counts) ** 2 for c in range(self.num_classes_))

    def _grow_tree(self, idx, num_samples_per_class, depth=0):

        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=self._gini(num_samples_per_class),
            num_samples=sum(num_samples_per_class),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if self.max_depth is not None:
            if depth >= self.max_depth:
                return node
            
        # pure node, no need to split
        if len(np.unique(self.y[idx])) == 1:
            return node
        
        feature_idx, thr, num_samples_per_class_left, num_samples_per_class_right = self._best_split(idx, num_samples_per_class)
        if feature_idx is not None:
            indices_left = idx[self.X[idx, feature_idx] <= thr]
            indices_right = idx[self.X[idx, feature_idx] > thr]
            node.feature_index = feature_idx
            node.threshold = thr
            node.left = self._grow_tree(indices_left, num_samples_per_class_left, depth + 1)
            node.right = self._grow_tree(indices_right, num_samples_per_class_right, depth + 1)
        return node

    def _best_split(self, idx, num_parent):
        """Find the best split for a node."""
        m = self.X[idx].shape[1]
        if m <= 1:  # only one feature
            return None, None

        best_gini = 1.0
        best_feature = None
        best_thr = None
        best_num_samples_per_class_left = None
        best_num_samples_per_class_right = None
        
        for f in range(m):

            thr = self.feature_mins[f] + (self.feature_maxs[f] - self.feature_mins[f])*np.random.rand()

            indices_left = idx[self.X[idx, f] <= thr]
            # indices_right = ~indices_left

            num_samples_per_class_left = [self.laplace_count(self.y[indices_left], i) for i in range(self.num_classes_)]
            num_samples_per_class_right = [max(num_parent[i] - num_samples_per_class_left[i], 0) for i in range(self.num_classes_)]

            num_left = sum(num_samples_per_class_left)
            num_right = sum(num_samples_per_class_right)

            gini_left = self._gini(num_samples_per_class_left)
            gini_right = self._gini(num_samples_per_class_right)

            gini = true_divide(num_left * gini_left + num_right * gini_right, num_left + num_right)

            if gini < best_gini:
                best_gini = gini
                best_feature = f
                best_thr = thr
                best_num_samples_per_class_left = num_samples_per_class_left
                best_num_samples_per_class_right = num_samples_per_class_right

            self.total_queries += 1

        return best_feature, best_thr, best_num_samples_per_class_left, best_num_samples_per_class_right

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y)."""
        self.num_classes_ = len(np.unique(y))

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        assert X.shape[1] == len(self.feature_mins)

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        self.X = X
        self.y = y

        np.random.seed(self.random_state)
        self.total_queries = 0

        num_samples_per_class = [self.laplace_count(y, i) for i in range(self.num_classes_)]
        idx = np.array(range(len(X)))
        self.tree_ = self._grow_tree(idx, num_samples_per_class)

        # delete data after training
        self.X = None
        self.y = None

    def laplace_count(self, y, value):
        # self.total_queries += 1
        count = np.sum(y == value)
        if self.epsilon is not None:
            return max(np.random.laplace(loc=count, scale=self.noise_scale), 1e-3)

        return count

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X):
        """Predict class for the input samples X."""

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y = [self._predict(inputs) for inputs in X]
        y = np.asarray(y)
        return y 
    

if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=ACC_SCALE, n_features=10, n_informative=5, n_redundant=0, random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_mins = [np.min(X[:, i]) for i in range(X.shape[1])]
    feature_maxs = [np.max(X[:, i]) for i in range(X.shape[1])]

    clf = DecisionTreeDP(max_depth=5, epsilon=None, feature_mins=feature_mins, feature_maxs=feature_maxs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    print("Max queries:", clf.max_queries)
    print("Total queries:", clf.total_queries)