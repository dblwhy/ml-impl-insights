import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, X, training=True):
        if training:
            # Create a mask: 1 with probability (1 - dropout_rate), 0 otherwise
            self.mask = (np.random.rand(*X.shape) > self.dropout_rate).astype(np.float32)
            return (X * self.mask) / (1 - self.dropout_rate)  # Scale the activations
        else:
            return X  # No dropout during inference

    def backward(self, d_out):
        """ Pass gradient only through the active neurons during training """
        return (d_out * self.mask) / (1 - self.dropout_rate)

# Example usage:
np.random.seed(42)  # For reproducibility
dropout_layer = Dropout(0.5)

X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print("Input:\n", X)

# Forward pass with dropout
X_dropped = dropout_layer.forward(X, training=True)
print("After Dropout (Training Mode):\n", X_dropped)

# Backward pass with dummy gradient
grad_input = dropout_layer.backward(np.ones_like(X_dropped))
print("Gradient Passed to Previous Layer:\n", grad_input)
