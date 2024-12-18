import csv
from config import config

class LinearRegressionScratch:
    def __init__(self, lr, epochs, regularization, reg_lambda):
        self.lr = lr
        self.epochs = epochs
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            y_predicted = [sum(x[i] * self.weights[i] for i in range(n_features)) + self.bias for x in X]
            errors = [y_predicted[i] - y[i] for i in range(n_samples)]

            for j in range(n_features):
                grad_w = sum(errors[i] * X[i][j] for i in range(n_samples)) / n_samples
                if self.regularization == 'l2':
                    grad_w += (self.reg_lambda / n_samples) * self.weights[j]
                elif self.regularization == 'l1':
                    grad_w += (self.reg_lambda / n_samples) * (1 if self.weights[j] > 0 else -1)

                self.weights[j] -= self.lr * grad_w

            grad_b = sum(errors) / n_samples
            self.bias -= self.lr * grad_b

    def predict(self, X):
        return [sum(x[i] * self.weights[i] for i in range(len(self.weights))) + self.bias for x in X]

    def mean_squared_error(self, y_true, y_pred):
        return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

    def r2_score(self, y_true, y_pred):
        mean_y = sum(y_true) / len(y_true)
        total_variance = sum((y - mean_y) ** 2 for y in y_true)
        explained_variance = sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_true)))
        return 1 - (explained_variance / total_variance)

model = LinearRegressionScratch(
    lr=config['learning_rate'],
    epochs=config['epochs'],
    regularization=config['regularization'],
    reg_lambda=config['reg_lambda']
)
