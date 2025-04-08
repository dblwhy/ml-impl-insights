import numpy as np

class L2Regularization:
    def __init__(self, lambda_value=0.01):
        self.lambda_value = lambda_value  # Regularization strength

    def compute_loss(self, loss, weights):
        """ Add L2 penalty to loss """
        return loss + self.lambda_value * np.sum(weights ** 2)

    def compute_gradient(self, gradient, weights):
        """ Apply L2 gradient update """
        return gradient + 2 * self.lambda_value * weights


# Example usage
l2 = L2Regularization(lambda_value=0.01)
loss = 0.5  # Example loss
weights = np.array([0.2, -0.5, 0.3])  # Example weights
gradient = np.array([0.1, -0.2, 0.1])  # Example gradient
new_loss = l2.compute_loss(loss, weights)
new_gradient = l2.compute_gradient(gradient, weights)

print("Updated Loss (with L2 penalty):", new_loss)
print("Updated Gradient (with L2):", new_gradient)
