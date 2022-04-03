import os
import re
from collections import defaultdict
from tkinter.tix import MAX

PARENT_PATH = '../data'
PATH = '../data/20news-bydate'
MAX_SENTENCE_LENGTH = 500
unknown_id = 1
padding_id = 0


def gen_data_and_vocab():
    def collect_data_from(parent_path, newsgroup_list, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            dir_path = f'{parent_path}/{newsgroup}'
            files = [(filename, f'{dir_path}/{filename}') 
                        for filename in os.listdir(dir_path)
                            if os.path.isfile(f'{dir_path}/{filename}')]
            files.sort()
            print(f'Processing: {group_id}-{newsgroup}')

            for filename, filepath in files:
                with open(filepath, errors='ignore') as f:
                    text = f.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(f'{group_id}<fff>{filename}<fff>{content}')
        return data

    word_count = defaultdict(int)
    dirs = [f'{PATH}/{d}' for d in os.listdir(PATH) if os.path.isdir(f'{PATH}/{d}')]

    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    newsgroup_list = [newsgroup for newsgroup in os.listdir(train_dir)]
    newsgroup_list.sort()

    train_data = collect_data_from(
        parent_path=train_dir,
        newsgroup_list=newsgroup_list,
        word_count=word_count
    )

    test_data = collect_data_from(
        parent_path=test_dir,
        newsgroup_list=newsgroup_list,
    )

    vocab = [word for word, count in word_count.items() if count > 10]
    vocab.sort()

    with open(f'{PARENT_PATH}/w2v/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    with open(f'{PARENT_PATH}/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))

    with open(f'{PARENT_PATH}/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))

def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = {word.rstrip(): word_id for word_id, word in enumerate(f, start=2)}

    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                        for line in f]

    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_SENTENCE_LENGTH]
        sentence_length = len(words)

        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(unknown_id))

        if len(words) < MAX_SENTENCE_LENGTH:
            num_padding = MAX_SENTENCE_LENGTH - len(words)
            encoded_text += [str(padding_id)] * num_padding

        encoded_data.append(f'{label}<fff>{doc_id}<fff>{sentence_length}<fff>' + ' '.join(encoded_text))

    dir_name = '/'.join(data_path.split('/')[:-1])
    file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(f'{dir_name}/{file_name}', 'w') as f:
        f.write('\n'.join(encoded_data))

if __name__ == '__main__':
    gen_data_and_vocab()

    encode_data(
        data_path=f'{PARENT_PATH}/w2v/20news-train-raw.txt',
        vocab_path=f'{PARENT_PATH}/w2v/vocab-raw.txt'
    )

    encode_data(
        data_path=f'{PARENT_PATH}/w2v/20news-test-raw.txt',
        vocab_path=f'{PARENT_PATH}/w2v/vocab-raw.txt'
    )