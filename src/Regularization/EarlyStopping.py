import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Initialize early stopping.
        :param patience: Number of epochs to wait before stopping if no improvement.
        :param min_delta: Minimum required improvement in loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if early stopping criteria is met.
        :param val_loss: Current validation loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # Reset counter if improvement occurs
        else:
            self.counter += 1  # Increment counter if no improvement

        if self.counter >= self.patience:
            self.early_stop = True  # Trigger early stopping
