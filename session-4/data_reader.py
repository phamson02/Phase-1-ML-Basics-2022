import numpy as np
from global_vars import *

class DataReader:
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size
        self.data, self.labels, self.sentence_lengths = self._load_data(data_path)
        self.num_epoch = 0
        self.batch_id = 0

    def _load_data(self, data_path):        
        with open(data_path) as f:
            doc_size = sum(1 for _ in f)

        data = np.zeros((doc_size, MAX_SENTENCE_LENGTH), dtype=np.int32)
        labels = np.empty(doc_size)
        sentence_lengths = np.empty(doc_size)

        with open(data_path) as f:
            for data_id, line in enumerate(f):
                features = line.split('<fff>')
                label = int(features[0])
                sentence_length = int(features[2])
                tokens = [int(i) for i in features[3].split()]

                data[data_id] = tokens
                labels[data_id] = label
                sentence_lengths[data_id] = sentence_length

        return data, labels, sentence_lengths
        
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
            self.data, self.labels, self.sentence_lengths = \
                self.data[indices], self.labels[indices], self.sentence_lengths[indices]

        return self.data[start:end], self.labels[start:end], self.sentence_lengths[start:end]

def load_datasets():
    train_data_reader = DataReader(
        data_path=f'{PATH}/20news-train-encoded.txt',
        batch_size=50,
    )

    test_data_reader = DataReader(
        data_path=f'{PATH}/20news-test-encoded.txt',
        batch_size=50,
    )

    return train_data_reader, test_data_reader

def get_vocab_size():
    with open(f'{PATH}/vocab-raw.txt') as f:
        vocab_size = sum(1 for _ in f)

    return vocab_size