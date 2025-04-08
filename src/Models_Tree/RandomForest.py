import numpy as np
from collections import Counter
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import GiniImpurity

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True):
        """
        Random Forest Classifier
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of trees in the forest.
        
        max_depth : int, default=None
            Maximum depth of each tree.
        
        min_samples_split : int, default=2
            The minimum number of samples required to split a node.
        
        max_features : int, float, str, default='sqrt'
            The number of features to consider when looking for the best split.
            If int, consider max_features features.
            If float, consider max_features * n_features features.
            If 'sqrt', consider sqrt(n_features) features.
            If 'log2', consider log2(n_features) features.
        
        bootstrap : bool, default=True
            Whether to use bootstrap samples when building trees.
        
        random_state : int, default=None
            Random seed for reproducibility.
        
        n_jobs : int, default=1
            The number of jobs to run in parallel. -1 means using all processors.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        # self.random_state = random_state
        # self.n_jobs = n_jobs
        self.estimators_ = []
        self.feature_importances_ = None
        
        # Set random state if provided
        # if random_state is not None:
        #     np.random.seed(random_state)

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y)."""
        n_samples, n_features = X.shape
        
        # Determine the number of jobs
        # n_jobs = self.n_jobs
        # if n_jobs
        # if n_jobs < 0:
        #     n_jobs = max(1, concurrent.futures.cpu_count() + 1 + n_jobs)
        # elif n_jobs == 0:
        #     n_jobs = 1

        # Create a list of random states for each tree
        # random_states = None
        # if self.random_state is not None:
        #     random_states = [self.random_state + i for i in range(self.n_estimators)]

        # Function to fit a single tree
        def fit_tree():
            # Create random state for this tree
            # tree_random_state = None
            # if random_states is not None:
            #     tree_random_state = random_states[i]

            # Create bootstrap sample
            # Ex. following code will choose 10 random values between 0-100 with replacement(=duplication)
            #   p.random.choice(100, 10, replace=True)
            #   => [37 25 77 72  9 20 72 69 79 47]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = GiniImpurity.ClassificationTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features, # fix GiniImpurity to support max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            return tree

        # Fit trees in parallel. We call this bootstrap aggregation, a.k.a bagging
        self.estimators_ = [fit_tree() for _ in range(self.n_estimators)]
        
        # Compute feature importances
        self.feature_importances_ = np.zeros(n_features)
        for tree in self.estimators_:
            self.feature_importances_ += tree.feature_importances_
        
        self.feature_importances_ /= len(self.estimators_)
        
        return self

    def predict(self, X):
        """
        Predict class for X using a majority vote of trees.
        X is a 2D array including bunch of data the caller wants to predict.
        """
        # Get predictions from all trees.
        #
        # Ex. returned value is a 2D array containing predicted values by each estimators
        #   predictions = [
        #                   [0 1], # predicted values by estimator 1
        #                   [1 1], # predicted values by estimator 2
        #                   [0 1], # predicted values by estimator 3
        #                 ]
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
        # Take the majority vote for each sample.
        # If user passes 10 samples, majority_votes is a 1D array with length 10
        majority_votes = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]): # iterate sample data times
            # Count the votes for each class
            sample_predictions = predictions[:, i] # Iterate through all rows in `predictions` and extract the value at index i
            counts = Counter(sample_predictions)
            # Get the class with the most votes
            majority_votes[i] = counts.most_common(1)[0][0]
        
        return majority_votes
    
    # def predict_proba(self, X):
    #     """Predict class probabilities for X."""
    #     # Get predictions from all trees
    #     predictions = np.array([tree.predict(X) for tree in self.estimators_])
        
    #     # Determine unique classes
    #     classes = np.unique(np.concatenate([np.unique(tree_pred) for tree_pred in predictions]))
    #     n_classes = len(classes)
        
    #     # Initialize probabilities
    #     proba = np.zeros((X.shape[0], n_classes))
        
    #     # Calculate probabilities for each sample
    #     for i in range(X.shape[0]):
    #         sample_predictions = predictions[:, i]
    #         counts = Counter(sample_predictions)
            
    #         # Calculate probabilities for each class
    #         for j, class_label in enumerate(classes):
    #             proba[i, j] = counts.get(class_label, 0) / self.n_estimators
        
    #     return proba, classes
    
    # def score(self, X, y):
    #     """Return the accuracy score of the predictions."""
    #     predictions = self.predict(X)
    #     return np.mean(predictions == y)
    
    # def __str__(self):
    #     """Generate a string representation of the random forest."""
    #     if not self.estimators_:
    #         return "Random Forest not fitted yet"
        
    #     result = [f"Random Forest with {len(self.estimators_)} trees:"]
    #     for i, tree in enumerate(self.estimators_[:3]):  # Show only first 3 trees
    #         result.append(f"\nTree {i+1}:")
    #         tree_str = str(tree).split('\n')
    #         result.extend([f"  {line}" for line in tree_str])
        
    #     if len(self.estimators_) > 3:
    #         result.append(f"\n... and {len(self.estimators_) - 3} more trees")
        
    #     return "\n".join(result)#
