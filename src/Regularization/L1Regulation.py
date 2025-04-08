import numpy as np

class L1Regularization:
    def __init__(self, lambda_value=0.01):
        self.lambda_value = lambda_value  # Regularization strength

    def compute_loss(self, loss, weights):
        """ Add L1 penalty to loss """
        return loss + self.lambda_value * np.sum(np.abs(weights))

    def compute_gradient(self, gradient, weights):
        """ Apply L1 gradient update (subgradient descent) """
        return gradient + self.lambda_value * np.sign(weights)

# Example usage
l1 = L1Regularization(lambda_value=0.01)
loss = 0.5  # Example loss
weights = np.array([0.2, -0.5, 0.3])  # Example weights
gradient = np.array([0.1, -0.2, 0.1])  # Example gradient

new_loss = l1.compute_loss(loss, weights)
new_gradient = l1.compute_gradient(gradient, weights)

print("Updated Loss (with L1 penalty):", new_loss)
print("Updated Gradient (with L1):", new_gradient)
