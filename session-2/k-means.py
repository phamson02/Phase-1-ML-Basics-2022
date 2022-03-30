import numpy as np
from collections import defaultdict

PATH = '../data/20news-bydate'

class Member:
    '''
    Represents a data point (document in the corpus) of the cluster.

    Attributes:
        _r_d: np.ndarray
            tf-idf vector of the document.
        _label: int
            newsgroup of the document.
        _doc_id: int
            name of the folder of the document.
    '''
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id

class Cluster:
    '''
    Represents a cluster

    Attributes:
        _centroid: np.ndarray
            centroid of the cluster.
        _members: list of Member
            list of members of the cluster.
    '''
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)

class Kmeans:
    '''
    

    Attributes:
        _num_clusters: int
            number of clusters.
        _clusters: list of Cluster
            list of clusters.
        _E: list of Member
            list of centroids.
        _S: float
            overall similarity.
        _data: list of Member
            list of all data points.
        _label_count: dict
            number of data points in each label.
    '''
    def __init__(self, num_clusters: int) -> None:
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(num_clusters)]
        self._E = []
        self._S = 0.0

    def load_data(self, data_path):
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

        self._data = []
        self._label_count = defaultdict(int)
        with open(f'{PATH}/{data_path}') as f:
            for line in f:
                features = line.split('<fff>')
                label, doc_id = int(features[0]), int(features[1])
                r_d = sparse_to_dense(features[2], vocab_size)
                self._label_count[label] += 1
                self._data.append(Member(r_d, label, doc_id))

    def _random_init(self, seed_value):
        np.random.seed(seed_value)
        picked_idx = np.random.choice(
                                    len(self._data),
                                    size=self._num_clusters, 
                                    replace=False,
                                )
        for i in range(self._num_clusters):
            self._clusters[i]._centroid = self._data[picked_idx[i]]._r_d

    def _compute_similarity(self, member, centroid):
        '''Calculate cosine similarity of a data point and a centroid'''
        return (member._r_d @ centroid) / (np.linalg.norm(member._r_d) * np.linalg.norm(centroid))

    def _select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self._compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity

    def _update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        ave_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(np.square(ave_r_d)))
        cluster._centroid = np.array(ave_r_d / sqrt_sum_sqr)
    
    def _stopping_condition(self, criterion, threshold):
        criterions = ['max_iters', 'centroid', 'similarity']
        assert criterion in criterions, f'criterion must be one of {criterions}, but got {criterion}'
        if criterion == 'max_iters':
            return self._iteration >= threshold

        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_changed = [centroid for centroid in E_new if centroid not in self._E]
            self._E = E_new
            return len(E_changed) <= threshold

        else:
            diff = self._new_S - self._S
            self._S = self._new_S
            return diff <= threshold

    def run(self, seed_value, criterion, threshold):
        self._random_init(seed_value)

        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            
            self._new_S = 0
            for member in self._data:
                max_S = self._select_cluster_for(member)
                self._new_S += max_S

            for cluster in self._clusters:
                self._update_centroid_of(cluster)

            self._iteration += 1

            if self._stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        majority_sum =0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            majority_label = max(set(member_labels), key=member_labels.count)
            majority_sum += member_labels.count(majority_label)
        return majority_sum / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_c, N = 0, 0, 0, len(self._data)
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            for label in set(member_labels):
                wk_cj = member_labels.count(label)
                wk = len(cluster._members)
                cj = self._label_count[label]
                I_value += (wk_cj / N) * np.log10(N * wk_cj / (wk * cj))
        
        for cluster in self._clusters:
            wk = len(cluster._members)
            H_omega += (wk / N) * np.log10(N / wk)
        
        for label in set(self._label_count):
            cj = self._label_count[label]
            H_c += (cj / N) * np.log10(N / cj)

        return I_value / ((H_omega + H_c) / 2)

if __name__ == '__main__':
    kmeans = Kmeans(num_clusters=20)
    kmeans.load_data('20news-full-tf-idf.txt')
    kmeans.run(seed_value=1, criterion='similarity', threshold=1e-4)
    print(kmeans.compute_purity())
    print(kmeans.compute_NMI())
    