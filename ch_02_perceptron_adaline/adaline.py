import numpy as np
from numpy import ndarray, float64


class AdalineGD(object):
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
            AdalineGD : object
        """

        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        breakpoint()
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)

        return self

    def net_input(self, X: ndarray) -> float64:
        """Calculate the net input

        Args:
            X (ndarray): Input vectors

        Returns:
            float64: Net input
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X: ndarray) -> ndarray:
        """Predict the label

        Args:
            X (ndarray): Input vectors

        Returns:
            ndarray: Label
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
