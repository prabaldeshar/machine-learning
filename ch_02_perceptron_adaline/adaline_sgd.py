import numpy as np
from numpy import ndarray, float64


class AdalineSGD(object):
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iter: int = 50,
        shuffle: bool = True,
        random_state: int = 1,
    ):
        """Initialize the object

        Args:
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            n_iter (int, optional): Passes over the training dataset. Defaults to 50.
            random_state (int, optional): Random number generator seed. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.weightsinitialized = False
        self.random_state = random_state

    def _initialize_weights(self, m):
        """Initialize weights to a small random number"""
        self.rgen = np.random.RandomState(self.random_state)
        self.weights = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.weightsinitialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.weights[1:] += self.learning_rate * xi.dot(error)
        self.weights[0] += self.learning_rate * error
        cost = (error**2) / 2
        return cost

    def _shuffle(self, X, y):
        """Shuffle the traning data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def fit(self, X: ndarray, y: ndarray):
        """Fit the training data

        Args:
            X (ndarray): shape [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_featutes is the number of features
            y (ndarray): shape [n_examples]
            Target values

        Returns:
            AdalineSGD : object
        """

        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """Fit the training data without reinitializing the weights"""
        if not self.weightsinitialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
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
