class CustomScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        n_features = len(X[0])
        self.means = [sum(row[i] for row in X) / len(X) for i in range(n_features)]
        self.stds = [
            (sum((row[i] - self.means[i]) ** 2 for row in X) / len(X)) ** 0.5
            for i in range(n_features)
        ]

    def transform(self, X):
        return [
            [(row[i] - self.means[i]) / self.stds[i] if self.stds[i] > 0 else 0 for i in range(len(row))]
            for row in X
        ]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)