import numpy as np
from collections import Counter
import queue

class XGBoostTree:
    def __init__(self, max_depth=6, min_samples_split=2, learning_rate=0.1, 
                 lambda_reg=1.0, gamma=0, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg  # L2 regularization
        self.gamma = gamma  # Minimum loss reduction for split
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None
        self.feature_importances_ = None
        
        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)

    def _sigmoid(self, x):
        """Sigmoid function for binary classification."""
        return 1 / (1 + np.exp(-x))
    
    def _calculate_residuals(self, y, y_pred):
        """Calculate gradients and hessians for binary logistic loss."""
        # Gradient: y_true - sigmoid(y_pred)
        gradients = y - self._sigmoid(y_pred)
        # Hessian: sigmoid(y_pred) * (1 - sigmoid(y_pred))
        hessians = self._sigmoid(y_pred) * (1 - self._sigmoid(y_pred))
        return gradients, hessians
    
    def fit(self, X, y, sample_weight=None, base_prediction=0.0):
        """Build a boosting tree using gradients and hessians."""
        n_samples, n_features = X.shape
        
        # Initialize prediction for first tree or default if not specified
        y_pred = np.full(n_samples, base_prediction)
        
        # Calculate initial gradients and hessians
        gradients, hessians = self._calculate_residuals(y, y_pred)
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # Determine max_features if not explicitly set
        if self.max_features is None:
            self.max_features = n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        
        # Create the root node
        root = {
            "X": X,
            "gradients": gradients,
            "hessians": hessians,
            "depth": 0,
            "is_leaf": False,
            "value": None,  # Will store the leaf weight
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None,
            "gain": 0.0,
            "n_samples": n_samples
        }
        
        # Use a queue to keep track of nodes to process
        node_queue = queue.Queue()
        node_queue.put(root)
        
        # Process nodes in breadth-first order
        while not node_queue.empty():
            node = node_queue.get()
            
            X_node = node["X"]
            gradients_node = node["gradients"]
            hessians_node = node["hessians"]
            depth = node["depth"]
            n_node_samples = len(gradients_node)
            
            # Calculate node weight (prediction value)
            sum_gradients = np.sum(gradients_node)
            sum_hessians = np.sum(hessians_node) + self.lambda_reg
            node_weight = - sum_gradients / sum_hessians
            
            # Check stopping criteria
            if (self.max_depth is not None and depth >= self.max_depth) or \
               n_node_samples < self.min_samples_split:
                # Make this a leaf node
                node["is_leaf"] = True
                node["value"] = node_weight
                # Clean up to save memory
                node.pop("X", None)
                node.pop("gradients", None)
                node.pop("hessians", None)
                continue
            
            # Find the best split
            best_feature, best_threshold, best_gain, best_left_mask, best_right_mask = self._best_split(
                X_node, gradients_node, hessians_node)
            
            # Check if the gain is significant enough
            if best_gain <= self.gamma:
                # Make this a leaf node
                node["is_leaf"] = True
                node["value"] = node_weight
                # Clean up to save memory
                node.pop("X", None)
                node.pop("gradients", None)
                node.pop("hessians", None)
                continue
            
            # Update feature importance
            self.feature_importances_[best_feature] += best_gain
            
            # Create masks for left and right splits
            left_mask = best_left_mask
            right_mask = best_right_mask
            
            # Skip if either split is empty
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                node["is_leaf"] = True
                node["value"] = node_weight
                # Clean up to save memory
                node.pop("X", None)
                node.pop("gradients", None)
                node.pop("hessians", None)
                continue
            
            # Set split information
            node["feature"] = best_feature
            node["threshold"] = best_threshold
            node["gain"] = best_gain
            
            # Create left child
            left_child = {
                "X": X_node[left_mask],
                "gradients": gradients_node[left_mask],
                "hessians": hessians_node[left_mask],
                "depth": depth + 1,
                "is_leaf": False,
                "value": None,
                "feature": None,
                "threshold": None,
                "left": None,
                "right": None,
                "gain": 0.0,
                "n_samples": np.sum(left_mask)
            }
            
            # Create right child
            right_child = {
                "X": X_node[right_mask],
                "gradients": gradients_node[right_mask],
                "hessians": hessians_node[right_mask],
                "depth": depth + 1,
                "is_leaf": False,
                "value": None,
                "feature": None,
                "threshold": None,
                "left": None,
                "right": None,
                "gain": 0.0,
                "n_samples": np.sum(right_mask)
            }
            
            # Link children to parent
            node["left"] = left_child
            node["right"] = right_child
            
            # Clean up parent node to save memory
            node.pop("X", None)
            node.pop("gradients", None)
            node.pop("hessians", None)
            
            # Add children to the queue
            node_queue.put(left_child)
            node_queue.put(right_child)
        
        # Normalize feature importances
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
            
        # Save the root as the tree
        self.tree = root
        return self

    def _best_split(self, X, gradients, hessians):
        """Find the best feature and threshold for splitting."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        n_samples, n_features = X.shape
        
        # Calculate node stats before split
        sum_grad = np.sum(gradients)
        sum_hess = np.sum(hessians)
        node_score = self._calc_node_score(sum_grad, sum_hess)
        
        # Randomly select subset of features to consider
        if self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)

        for feature in feature_indices:
            X_feature = X[:, feature]
            
            # Sort values for this feature
            sorted_indices = np.argsort(X_feature)
            sorted_X = X_feature[sorted_indices]
            sorted_gradients = gradients[sorted_indices]
            sorted_hessians = hessians[sorted_indices]
            
            # Get unique values
            unique_values = np.unique(sorted_X)
            if len(unique_values) <= 1:
                continue
            
            # Create thresholds midway between unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            # Calculate cumulative sums for efficient split evaluation
            grad_left = 0
            hess_left = 0
            
            # Evaluate all possible splits
            for i, threshold in enumerate(thresholds):
                # Sum gradients and hessians for samples up to index i
                while i < len(sorted_X) and sorted_X[i] <= threshold:
                    grad_left += sorted_gradients[i]
                    hess_left += sorted_hessians[i]
                    i += 1
                
                # If either child would be empty, skip
                if grad_left == 0 or grad_left == sum_grad:
                    continue
                
                # Calculate gradients and hessians for right child
                grad_right = sum_grad - grad_left
                hess_right = sum_hess - hess_left
                
                # Calculate gain
                gain = self._calc_split_gain(grad_left, hess_left, grad_right, hess_right, node_score)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_mask = X[:, feature] <= threshold
                    best_right_mask = ~best_left_mask

        return best_feature, best_threshold, best_gain, best_left_mask, best_right_mask

    def _calc_node_score(self, sum_gradients, sum_hessians):
        """Calculate score for a node based on its gradients and hessians."""
        if sum_hessians == 0:
            return 0
        return (sum_gradients ** 2) / (sum_hessians + self.lambda_reg)
    
    def _calc_split_gain(self, grad_left, hess_left, grad_right, hess_right, parent_score):
        """Calculate the gain for a potential split."""
        # Calculate score for left node
        left_score = self._calc_node_score(grad_left, hess_left)
        
        # Calculate score for right node
        right_score = self._calc_node_score(grad_right, hess_right)
        
        # Calculate gain
        gain = 0.5 * (left_score + right_score - parent_score)
        
        return gain

    def predict(self, X):
        """Predict raw scores for X."""
        return np.array([self._predict_single(x) * self.learning_rate for x in X])
    
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        scores = self.predict(X)
        probas = self._sigmoid(scores)
        return np.vstack([1 - probas, probas]).T
    
    def predict_class(self, X):
        """Predict class labels for X."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def _predict_single(self, x):
        """Predict score for a single sample x using iterative tree traversal."""
        node = self.tree
        
        # Traverse the tree iteratively until reaching a leaf node
        while not node["is_leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
                
        return node["value"]

    def __str__(self):
        """Generate a string representation of the tree using iterative approach."""
        if self.tree is None:
            return "Tree not fitted yet"
        
        result = []
        
        # Use stack for depth-first traversal
        # Each entry is a tuple: (node, indent, is_right_child)
        stack = [(self.tree, "", False)]
        
        while stack:
            node, indent, is_right_child = stack.pop()
            
            if node["is_leaf"]:
                result.append(f"{indent}Leaf: weight = {node['value']:.4f}")
            else:
                feature = node["feature"]
                threshold = node["threshold"]
                gain = node["gain"]
                
                if is_right_child:
                    result.append(f"{indent}Feature {feature} > {threshold:.4f} (gain={gain:.4f})")
                else:
                    result.append(f"{indent}Feature {feature} <= {threshold:.4f} (gain={gain:.4f})")
                
                # Add right child to stack first (so left gets processed first)
                if node["right"] is not None:
                    stack.append((node["right"], indent + "  ", True))
                
                # Add left child to stack
                if node["left"] is not None:
                    stack.append((node["left"], indent + "  ", False))
        
        return "\n".join(result)


class XGBoost:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 lambda_reg=1.0, gamma=0, max_features=None, random_state=None, n_jobs=1):
        """
        XGBoost Classifier for binary classification
        
        Parameters:
        -----------
        n_estimators : int, default=100
            The number of boosting stages to perform.
        
        max_depth : int, default=6
            Maximum depth of each tree.
        
        learning_rate : float, default=0.1
            Step size shrinkage used to prevent overfitting.
        
        lambda_reg : float, default=1.0
            L2 regularization term on weights.
        
        gamma : float, default=0
            Minimum loss reduction required to make a further partition.
        
        max_features : int, float, str, default=None
            The number of features to consider when looking for the best split.
        
        random_state : int, default=None
            Random seed for reproducibility.
        
        n_jobs : int, default=1
            Number of parallel threads used to run xgboost.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.estimators_ = []
        self.feature_importances_ = None
        self.base_score = 0.0  # Initial prediction
        
        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)

    def _sigmoid(self, x):
        """Sigmoid function for binary classification."""
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """Build a boosted model from the training set (X, y)."""
        # Convert target to 0/1 for binary classification
        y = np.array(y, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        # Initialize predictions with base score (log-odds of class 1)
        if self.base_score == 0.0:
            pos_count = np.sum(y == 1)
            neg_count = n_samples - pos_count
            if pos_count > 0 and neg_count > 0:
                self.base_score = np.log(pos_count / neg_count)
        
        # Current predictions (in log-odds space)
        pred = np.full(n_samples, self.base_score)
        
        # Fit trees sequentially
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Create new tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                lambda_reg=self.lambda_reg,
                gamma=self.gamma,
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Fit tree to current residuals
            tree.fit(X, y, base_prediction=pred)
            
            # Update predictions
            update = tree.predict(X)
            pred += update
            
            # Store tree
            self.estimators_.append(tree)
        
        # Compute feature importances
        self.feature_importances_ = np.zeros(n_features)
        for tree in self.estimators_:
            self.feature_importances_ += tree.feature_importances_
        
        self.feature_importances_ /= len(self.estimators_)
        
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        # Sum contributions from all trees
        scores = np.full(X.shape[0], self.base_score)
        
        for tree in self.estimators_:
            scores += tree.predict(X)
        
        # Convert log-odds to probabilities
        probas = self._sigmoid(scores)
        
        # Return probabilities for both classes
        return np.vstack([1 - probas, probas]).T

    def predict(self, X):
        """Predict class for X."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X, y):
        """Return the accuracy score of the predictions."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def __str__(self):
        """Generate a string representation of the boosted model."""
        if not self.estimators_:
            return "XGBoost model not fitted yet"
        
        result = [f"XGBoost model with {len(self.estimators_)} trees:"]
        result.append(f"Base score: {self.base_score:.4f}")
        
        # Show details of first few trees
        for i, tree in enumerate(self.estimators_[:3]):
            result.append(f"\nTree {i+1}:")
            tree_str = str(tree).split('\n')
            result.extend([f"  {line}" for line in tree_str])
        
        if len(self.estimators_) > 3:
            result.append(f"\n... and {len(self.estimators_) - 3} more trees")
        
        return "\n".join(result)
