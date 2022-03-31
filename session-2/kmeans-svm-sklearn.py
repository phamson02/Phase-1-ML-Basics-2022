import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC

PATH = '../data/20news-bydate'

def load_data(data_path):
    with open(f'{PATH}/words-idfs.txt') as f:
        vocab_size = sum(1 for _ in f)

    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = np.zeros(vocab_size)
        indices_tfidf = sparse_r_d.split()
        for index_tfidf in indices_tfidf:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return r_d

    data = np.loadtxt(
        fname=data_path,
        delimiter='<fff>',
        usecols=2,
        converters={
            2: lambda x: sparse_to_dense(str(x)[2:-1], vocab_size)
        },
    )

    labels = np.loadtxt(
        fname=data_path,
        delimiter='<fff>',
        usecols=0,
    )

    return data, labels

def compute_accuracy(y_pred, y_test):
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def clustering_with_KMeans():
    X, _ = load_data(data_path=f'{PATH}/20news-full-tf-idf.txt')
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
    X_train, y_train = load_data(data_path=f'{PATH}/20news-train-tf-idf.txt')
    classifier = LinearSVC(
        C=10.0,
        tol=1e-3,
        verbose=False,
    )
    classifier.fit(X_train, y_train)

    X_test, y_test = load_data(data_path=f'{PATH}/20news-test-tf-idf.txt')
    y_pred = classifier.predict(X_test)
    accuracy = compute_accuracy(y_pred, y_test)

    return accuracy

def classifying_with_kernel_SVM():
    X_train, y_train = load_data(data_path=f'{PATH}/20news-train-tf-idf.txt')
    print(X_train)
    print(y_train)
    classifier = SVC(
        C=50.0,
        kernel='rbf',
        gamma=0.1,
        tol=1e-3,
        verbose=False,
    )
    classifier.fit(X_train, y_train)


    X_test, y_test = load_data(data_path=f'{PATH}/20news-test-tf-idf.txt')
    y_pred = classifier.predict(X_test)
    accuracy = compute_accuracy(y_pred, y_test)

    return accuracy


if __name__ == '__main__':
    # # accuracy_with_KMeans = clustering_with_KMeans()
    # accuracy_with_linear_SMSVC = classifying_with_linear_SVM()
    # accuracy_with_kernel_SMSVC = classifying_with_kernel_SVM()
    # # print(f'accuracy_with_KMeans: {accuracy_with_KMeans}')
    # print(f'accuracy_with_linear_SMSVC: {accuracy_with_linear_SMSVC}')
    # print(f'accuracy_with_kernel_SMSVC: {accuracy_with_kernel_SMSVC}')
    data, labels = load_data(data_path=f'{PATH}/20news-full-tf-idf.txt')
    print(data[0,-2])
    
