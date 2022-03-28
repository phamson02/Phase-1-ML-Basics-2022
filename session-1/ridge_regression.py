from pathlib import Path
import numpy as np


def get_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append([float(i) for i in line.split()])

    data = np.array(data)
    X = data[:, :-1]  # attributes from A1 -> A15
    Y = data[:, -1]  # attribute B
    return X, Y


def normalize_and_add_ones(X):
    X = np.array(X)
    X_min = np.array([[np.amin(X[:, id_col])
                       for id_col in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_max = np.array([[np.amax(X[:, id_col])
                       for id_col in range(X.shape[1])]
                      for _ in range(X.shape[0])])
    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.ones((X.shape[0], 1))
    return np.concatenate((ones, X_normalized), axis=1)


class RidgeRegression:

    def __init__(self):
        pass

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]
        self.LAMBDA = LAMBDA

        self.w = np.linalg.inv(
            X_train.T @ X_train + LAMBDA * np.eye(X_train.shape[1])
        ) @ X_train.T @ Y_train

    def fit_gradient_descent(self, X_train, Y_train, LAMBDA, alpha, max_iter=100, batch_size=120):
        self.w = np.random.rand(X_train.shape[1])
        last_loss = 10e+8
        for iter in range(max_iter):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            X_train = X_train[arr]
            Y_train = Y_train[arr]
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            for i in range(total_minibatch):
                index = i * batch_size
                X_batch = X_train[index:index + batch_size]
                Y_batch = Y_train[index:index + batch_size]
                grad = X_batch.T @ (X_batch @ self.w -
                                    Y_batch) + LAMBDA * self.w
                self.w = self.w - alpha * grad
            new_loss = self.computeRSS(X_train, Y_train)
            if abs(new_loss - last_loss) < 1e-5:
                break
            last_loss = new_loss

    def predict(self, X_test):
        return np.dot(X_test, self.w)

    def computeRSS(self, X_test, Y_test):
        return np.sum((self.predict(X_test) - Y_test) ** 2) / Y_test.shape[0]

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            total_RSS = 0
            valid_ids = [range(i * X_train.shape[0] // num_folds, (i+1) * X_train.shape[0] // num_folds)
                         for i in range(num_folds-1)]
            valid_ids.append(
                range((num_folds-1) * X_train.shape[0] // num_folds, X_train.shape[0]))
            train_ids = [[k for k in range(
                X_train.shape[0]) if k not in valid_ids[i]] for i in range(num_folds)]
            for i in range(num_folds):
                X_valid_fold = X_train[valid_ids[i]]
                Y_valid_fold = Y_train[valid_ids[i]]
                X_train_fold = X_train[train_ids[i]]
                Y_train_fold = Y_train[train_ids[i]]
                self.fit(X_train_fold, Y_train_fold, LAMBDA)
                total_RSS += self.computeRSS(X_valid_fold, Y_valid_fold)
            return total_RSS / num_folds

        np.split

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    minimum_RSS = aver_RSS
                    best_LAMBDA = current_LAMBDA
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(
            best_LAMBDA=0,
            minimum_RSS=10e+8,
            LAMBDA_values=range(50),
        )

        LAMDA_values = [
            k/1000 for k in range(max(0, (best_LAMBDA - 1)*1000), (best_LAMBDA + 1)*1000)]

        best_LAMBDA, minimum_RSS = range_scan(
            best_LAMBDA=best_LAMBDA,
            minimum_RSS=minimum_RSS,
            LAMBDA_values=LAMDA_values,
        )

        return best_LAMBDA


if __name__ == '__main__':
    X, Y = get_data(Path('../data/death-rates/x28-data.txt'))
    X = normalize_and_add_ones(X)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print(f'Best LAMBDA found: {best_LAMBDA}')
    ridge_regression.fit(X_train, Y_train, best_LAMBDA)

    RSS = ridge_regression.computeRSS(X_test, Y_test)
    print(f'RSS: {RSS}')
