import numpy as np

# noinspection PyPep8Naming
class LinearSVM:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000, random_state=0):
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.bias = None
        self.weights = None

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        _, n_features = X.shape

        self.bias = np.zeros(n_classes)
        self.weights = np.zeros((n_classes, n_features))
        np.random.seed(self.random_state)

        for i, cls in enumerate(classes):
            y_binary = np.where(y == cls, 1, -1)
            self.bias[i], self.weights[i] = self._find_weights(X, y_binary)

    def _find_weights(self, X, y):
        n_samples, n_features = X.shape
        bias = 0
        weights = np.random.randn(n_features)

        for _ in range(self.epochs):
            for i in range(n_samples):
                y_pred = X[i] @ weights + bias
                margin = y[i] * y_pred

                if margin < 1:
                    dw = -self.C * y[i] * X[i]
                    db = -self.C * y[i]

                    weights -= self.learning_rate * dw
                    bias -= self.learning_rate * db

        return bias, weights


    def predict(self, X):
        scores = X @ self.weights.T + self.bias
        return np.argmax(scores, axis=1)


# noinspection PyPep8Naming
class KernelSVM:
    def __init__(self, C=1.0, gamma=0.1, learning_rate=0.01, epochs=1000, random_state=0):
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.alpha = None
        self.bias = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.K = None

    def _rbf_kernel(self, X1, X2):
        return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)

    def fit(self, X, y):
        n_samples = X.shape[0]
        np.random.seed(self.random_state)

        self.K = self._rbf_kernel(X, X)
        self.alpha = np.zeros(n_samples)

        classes = np.unique(y)
        y = np.where(y == classes[1], 1, -1)

        for _ in range(self.epochs):
            for i in range(n_samples):
                decision_function = np.sum(self.alpha * y * self.K[:, i]) + self.bias
                margin = y[i] * decision_function

                if margin < 1:
                    self.alpha[i] += self.learning_rate * (self.C - margin)
                    self.bias += self.learning_rate * y[i]

        support_vector_indices = self.alpha > 1e-5
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.alpha = self.alpha[support_vector_indices]

    def predict(self, X):
        K_test = self._rbf_kernel(X, self.support_vectors)
        decision_function = np.sum(self.alpha * self.support_vector_labels * K_test, axis=1) + self.bias
        return np.where(decision_function > 0, 1, 0)