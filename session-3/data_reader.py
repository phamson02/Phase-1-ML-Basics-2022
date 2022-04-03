import numpy as np

PATH = '../data/20news-bydate'

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        self.data, self.labels = self._load_data(data_path)
        self.num_epoch = 0
        self.batch_id = 0

    def _load_data(self, data_path):
        with open(f'{PATH}/words-idfs.txt') as f:
            vocab_size = sum(1 for _ in f)
        
        with open(data_path) as f:
            doc_size = sum(1 for _ in f)

        data = np.empty((doc_size, vocab_size))
        labels = np.empty(doc_size)

        with open(data_path) as f:
            for data_id, line in enumerate(f):
                r_d = np.zeros(vocab_size)
                features = line.split('<fff>')
                label = int(features[0])
                tokens = features[2].split()
                for token in tokens:
                    idx, value = int(token.split(':')[0]), float(token.split(':')[1])
                    r_d[idx] = value
                data[data_id] = r_d
                labels[data_id] = label

        return data, labels
        
    def next_batch(self):
        start = self.batch_id * self._batch_size
        end = start + self._batch_size
        self.batch_id += 1

        if end + self._batch_size > self.data.shape[0]:
            end = self.data.shape[0]
            self.num_epoch += 1
            self.batch_id = 0
            indices = np.arange(self.data.shape[0])
            np.random.seed(42)
            np.random.shuffle(indices)
            self.data, self.labels = self.data[indices], self.labels[indices]

        return self.data[start:end], self.labels[start:end]

def load_datasets():
    train_data_reader = DataReader(
        data_path=f'{PATH}/20news-train-tf-idf.txt',
        batch_size=50,
    )

    test_data_reader = DataReader(
        data_path=f'{PATH}/20news-test-tf-idf.txt',
        batch_size=50,
    )

    return train_data_reader, test_data_reader
        