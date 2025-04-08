import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.nansum((x1 - x2) ** 2))  # Ignore NaN values in distance computation

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances (ignoring NaNs)
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote (for classification)
        if isinstance(self.y_train[0], (str, int)):  
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        # Mean value (for regression)
        else:
            return np.mean(k_nearest_labels)
    
    def knn_impute(self, X):
        """ Fill missing values using KNN """
        X_imputed = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    # Predict missing value using KNN
                    x_complete = np.delete(X[i], j)  # Exclude missing feature
                    train_complete = np.delete(X, j, axis=1)  # Exclude column from training
                    non_nan_indices = ~np.isnan(train_complete).any(axis=1)  # Consider only non-NaN rows
                    knn = KNN(k=self.k)
                    knn.fit(train_complete[non_nan_indices], X[non_nan_indices, j])  # Fit on non-missing values
                    X_imputed[i, j] = knn._predict(x_complete)
        return X_imputed

# Sample data with missing values (NaN)
X_train = np.array([
    [2.0, np.nan, 3.0],
    [1.0, 2.0, np.nan],
    [4.0, 1.0, 5.0],
    [np.nan, 3.0, 2.0]
])

y_train = np.array([0, 1, 0, 1])  # Example labels (for classification)

# Create KNN instance
knn = KNN(k=2)

# Fill missing values using KNN imputation
X_filled = knn.knn_impute(X_train)

print("Filled Data:\n", X_filled)

# Train KNN classifier
knn.fit(X_filled, y_train)

# Example test sample
X_test = np.array([[2.5, 2.0, 3.5]])

# Predict
prediction = knn.predict(X_test)
print("Predicted Class:", prediction)
