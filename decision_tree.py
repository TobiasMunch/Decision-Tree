import numpy as np


class Node:
    """A single node in the tree. Leaf nodes have `value` set; internal
    nodes have `feature_index`/`threshold`/`left`/`right` set instead."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        if criterion not in ("entropy", "gini"):
            raise ValueError('criterion must be "entropy" or "gini"')
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    # ---------- public API ----------

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0, prefix="root: "):
        """Human-readable dump of the tree for debugging."""
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}{prefix}leaf -> class {node.value}")
        else:
            print(f"{indent}{prefix}X[{node.feature_index}] <= {node.threshold:.4f} ?")
            self.print_tree(node.left, depth + 1, prefix="True  -> ")
            self.print_tree(node.right, depth + 1, prefix="False -> ")

    # ---------- impurity ----------

    def _impurity_from_counts(self, counts):
        """Impurity computed straight from class counts.

        `counts` has shape (..., n_classes); the impurity is computed over the
        last axis, so this works both on a single count vector and on a whole
        stack of candidate splits at once.
        """
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=-1, keepdims=True)
        probs = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

        if self.criterion == "entropy":
            # p * log2(p) -> 0 as p -> 0, but log2(0) warns, so substitute a
            # dummy 1.0 wherever p == 0 (then p * log2(1) = 0 anyway).
            safe = np.where(probs > 0, probs, 1.0)
            return -np.sum(probs * np.log2(safe), axis=-1)
        # gini
        return 1.0 - np.sum(probs ** 2, axis=-1)

    def _impurity(self, y):
        counts = np.unique(y, return_counts=True)[1]
        return self._impurity_from_counts(counts)

    # ---------- splitting ----------

    def _best_split(self, X, y):
        """Find the (feature, threshold) pair with the highest information gain.

        Every midpoint between consecutive distinct values of a feature is a
        candidate threshold. Rather than re-scanning the data once per
        candidate, the samples are sorted by the feature once and a cumulative
        sum of one-hot class labels gives the class counts on the left of every
        possible cut position at the same time. All candidate thresholds for a
        feature are then scored in a single vectorised expression, which makes
        the cost per feature O(n log n) (dominated by the sort) instead of
        O(n * n_thresholds).
        """
        n_samples, n_features = X.shape
        if n_samples < 2:
            return None, None, -np.inf

        # integer-code the labels so they can be one-hot encoded
        y_codes = np.unique(y, return_inverse=True)[1]
        n_classes = y_codes.max() + 1
        onehot = np.zeros((n_samples, n_classes))
        onehot[np.arange(n_samples), y_codes] = 1.0

        parent_impurity = self._impurity_from_counts(onehot.sum(axis=0))

        # number of samples on each side for a cut after position i (i = 0..n-2)
        n_left = np.arange(1, n_samples, dtype=float)
        n_right = n_samples - n_left

        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            values = X[:, feature_index]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]

            lower, upper = sorted_values[:-1], sorted_values[1:]

            # a cut is only meaningful between two *different* values; the
            # second condition guards the rare case where the midpoint rounds
            # up to the upper value, which would make the scored partition
            # disagree with the `<=` test used at predict time
            thresholds = (lower + upper) / 2.0
            valid = (lower < upper) & (thresholds < upper)
            if not valid.any():
                continue  # constant feature at this node, nothing to split on

            # cumulative class counts: row i = counts among the first i+1 samples
            cumulative = np.cumsum(onehot[order], axis=0)
            left_counts = cumulative[:-1]
            right_counts = cumulative[-1] - left_counts

            weighted_child_impurity = (
                (n_left / n_samples) * self._impurity_from_counts(left_counts)
                + (n_right / n_samples) * self._impurity_from_counts(right_counts)
            )
            gains = np.where(valid, parent_impurity - weighted_child_impurity, -np.inf)

            candidate = int(np.argmax(gains))
            if gains[candidate] > best_gain:
                best_gain = gains[candidate]
                best_feature = feature_index
                best_threshold = thresholds[candidate]

        return best_feature, best_threshold, best_gain

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ---------- recursive tree building ----------

    def _build_tree(self, X, y, depth):
        # stopping condition 1: pure node
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # stopping condition 2: identical feature vectors, can't split further
        if np.all(X == X[0], axis=0).all():
            return Node(value=self._most_common_label(y))

        # practical stopping condition: max_depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._most_common_label(y))

        feature_index, threshold, gain = self._best_split(X, y)

        # no feature gives a useful split
        if feature_index is None or gain <= 0:
            return Node(value=self._most_common_label(y))

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_child, right=right_child)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._predict_one(x, branch)


def permutation_importance(model, X, y, metric, n_repeats=30, seed=None):
    """Permutation feature importance.

    Follows the outline in the scikit-learn user guide:

    Paraphrase from: https://scikit-learn.org/stable/modules/permutation_importance.html#outline-of-the-permutation-importance-algorithm

        1. Compute the reference score s of the fitted model on D = (X, y).
        2. For each feature j, and each repetition k in 1..K:
             - shuffle column j of D to make a corrupted dataset D~(k, j)
             - compute the score s(k, j) of the model on D~(k, j)
        3. importance_j = s - (1/K) * sum_k s(k, j)

    Parameters
    ----------
    model : fitted model exposing .predict(X)
    X, y  : the data to evaluate on (a held-out set is preferable to the
            training set)
    metric : callable metric(y_true, y_pred) -> float, higher is better
    n_repeats : number of shuffles per feature (K above)
    seed : seed for the random number generator, for reproducibility

    Returns
    -------
    dict with keys:
        "importances"      : (n_features, n_repeats) array of per-repeat drops
        "importances_mean" : (n_features,) mean drop per feature
        "importances_std"  : (n_features,) std of the drop per feature
        "reference_score"  : the unpermuted score s
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    n_features = X.shape[1]

    # step 1: reference score on the untouched data
    reference_score = metric(y, model.predict(X))

    importances = np.zeros((n_features, n_repeats))

    for j in range(n_features):
        original_column = X[:, j].copy()

        for k in range(n_repeats):
            # step 2: corrupt only column j, leaving every other column intact
            X[:, j] = rng.permutation(original_column)
            permuted_score = metric(y, model.predict(X))
            importances[j, k] = reference_score - permuted_score

        # restore the column before moving on to the next feature
        X[:, j] = original_column

    return {
        "importances": importances,
        "importances_mean": importances.mean(axis=1),
        "importances_std": importances.std(axis=1),
        "reference_score": reference_score,
    }


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=3000, n_features=5,n_redundant=0, n_informative=3,
        n_classes=2, random_state=42, shuffle=False
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tree = DecisionTree(criterion="entropy", max_depth=5)
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)

    accuracy = (preds == y_test).mean()
    print(f"Test accuracy: {accuracy:.3f}\n")
    tree.print_tree()
    result = permutation_importance(
        tree, X_test, y_test,
        metric=lambda y_true, y_pred: (y_true == y_pred).mean(),
        n_repeats=10,
        seed=42,
    )
    print(result["importances_mean"])