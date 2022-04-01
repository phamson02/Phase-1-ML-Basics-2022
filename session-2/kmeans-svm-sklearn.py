import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC

PATH = '../data/20news-bydate'

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = np.zeros(vocab_size)
        indices_tfidf = sparse_r_d.split()
        for index_tfidf in indices_tfidf:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return r_d

    with open(f'{PATH}/words-idfs.txt') as f:
        vocab_size = sum(1 for _ in f)

    with open(f'{PATH}/{data_path}') as f:
        data_size = sum(1 for _ in f)

    data = np.empty((data_size, vocab_size))
    labels = np.empty(data_size)
    with open(f'{PATH}/{data_path}') as f:
        for idx, line in enumerate(f):
            features = line.split('<fff>')
            label = int(features[0])    
            r_d = sparse_to_dense(features[2], vocab_size)
            data[idx] = r_d
            labels[idx] = label

    return data, labels

def compute_accuracy(classifier):
    X_test, y_test = load_data(data_path=f'20news-test-tf-idf.txt')
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def clustering_with_KMeans():
    X, _ = load_data(data_path=f'20news-full-tf-idf.txt')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2022,
    )
    kmeans.fit(X)

    return kmeans

def classifying_with_linear_SVM():
    X_train, y_train = load_data(data_path=f'20news-train-tf-idf.txt')
    classifier = LinearSVC(
        C=10.0,
        tol=1e-3,
        verbose=False,
    )
    classifier.fit(X_train, y_train)

    return classifier

def classifying_with_kernel_SVM():
    X_train, y_train = load_data(data_path=f'20news-train-tf-idf.txt')
    classifier = SVC(
        C=50.0,
        kernel='rbf',
        gamma=0.1,
        tol=1e-3,
        verbose=False,
    )
    classifier.fit(X_train, y_train)

    return classifier


if __name__ == '__main__':
    linear_SMSVC = classifying_with_linear_SVM()
    kernel_SMSVC = classifying_with_kernel_SVM()
    print(f'accuracy_with_linear_SMSVC: {compute_accuracy(linear_SMSVC)}')
    print(f'accuracy_with_kernel_SMSVC: {compute_accuracy(kernel_SMSVC)}')