import numpy as np
from numpy import ndarray, float64


class Perceptron(object):
    def __init__(
        self, learning_rate: float = 0.01, n_iter: int = 50, random_state: int = 1
    ):
        """Initialize the object

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            n_iter (int, optional): Passes over the training dataset. Defaults to 50.
            random_state (int, optional): Random number generator seed. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: ndarray, y: ndarray):
        """Fit the training data

        Args:
            X (ndarray): shape [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_featutes is the number of features
            y (ndarray): shape [n_examples]
            Target values

        Returns:
            Perceptron: object
        """
        breakpoint()
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return self

    def net_input(self, X: ndarray) -> float64:
        """Calculate the net input

        Args:
            X (ndarray): Input vectors

        Returns:
            float64: Net input
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X: ndarray) -> ndarray:
        """Predict the label

        Args:
            X (ndarray): Input vectors

        Returns:
            ndarray: Label
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
