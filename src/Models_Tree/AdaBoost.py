import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class DecisionStump:
    """Weak classifier used in AdaBoost."""
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, sample_weights):
        """Train a decision stump."""
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_i in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature_i]))
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * X[:, feature_i] < polarity * threshold] = -1

                    weighted_error = np.sum(sample_weights[y != predictions])

                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.feature_index = feature_i
                        self.threshold = threshold
                        self.polarity = polarity

        self.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

    def predict(self, X):
        """Predict using the trained decision stump."""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        predictions[self.polarity * X[:, self.feature_index] < self.polarity * self.threshold] = -1
        return predictions

class AdaBoost:
    """AdaBoost implementation."""
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        """Train AdaBoost."""
        n_samples, _ = X.shape
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, sample_weights)
            predictions = stump.predict(X)

            error = np.sum(sample_weights[y != predictions])

            sample_weights *= np.exp(-stump.alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.models.append(stump)

    def predict(self, X):
        """Make predictions using the trained AdaBoost model."""
        final_predictions = np.zeros(X.shape[0])
        for stump in self.models:
            final_predictions += stump.alpha * stump.predict(X)
        return np.sign(final_predictions)

# Generate toy dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels to -1, 1 for AdaBoost

# Train AdaBoost
adaboost = AdaBoost(n_estimators=10)
adaboost.fit(X, y)

# Predict and visualize decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = adaboost.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.title("AdaBoost Decision Boundary")
plt.show()
