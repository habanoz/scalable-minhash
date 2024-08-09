from heapq import heapify, heappushpop
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import numpy.typing as npt
import tqdm

class PairCache:
    def __init__(self, limit=1000):
        self.limit = limit
        self.h = []
        self.push_counts = 0

    @property
    def size(self):
        return len(self.h)

    def add(self, i, j, score):
        if self.limit is None or self.size < self.limit-1:
            self.h.append((score, i, j))
        elif self.size == self.limit-1:
            self.h.append((score, i, j))
            heapify(self.h)  # O(N)
        else:
            if self.h[0][0] < score:  # h[0] is the smallest value in the min heap
                heappushpop(self.h, (score, i, j))  # O(logN)
                self.push_counts += 1

    def get_list(self):
        self.h.sort(reverse=True)
        return list(map(lambda x: (x[1], x[2], x[0].item()), self.h))


class MinHash:
    def __init__(self, n_hashes, cache_size=1000, seed=123):
        self.cv = None
        rng = np.random.default_rng(seed)
        self.cv = CountVectorizer(binary=True, ngram_range=(1, 1), analyzer="word")

        # sympy.nextprime(self.max_token_value)
        max_token_value = 2 ** 32 - 1
        self.next_prime = 4294967311
        
        self.n_hashes = n_hashes
        self.cache_size = cache_size

        self.A = rng.integers(0, max_token_value, n_hashes, dtype=np.int64)
        self.B = rng.integers(0, max_token_value, n_hashes, dtype=np.int64)

    def _fit(self, corpus):
        self.cv.fit(corpus)

    def _sparse_vectorize(self, text_list):
        return self.cv.transform(text_list).tolil().rows

    def _build_signatures(self, sparse_vector)->npt.NDArray:
        return np.asarray([self._create_signature(v) for v in sparse_vector])
    
    def _create_signature(self, sparse_vector)->npt.NDArray:
        signature = np.matmul(np.asarray(sparse_vector).reshape(-1, 1), self.A.reshape(1, -1))
        signature += self.B
        signature %= self.next_prime
        signature = signature.min(axis=0)
        return signature

    def _create_pairs(self, signatures:npt.NDArray):
        pair_cache = PairCache(limit=self.cache_size)
        N, H = signatures.shape

        for i in tqdm.tqdm(range(N)):
            # broadcast and subtract from the remaining of the matrix
            matches = (signatures[i]-signatures[i+1:, :]) == 0
            scores = matches.sum(axis=1) / H

            for j, score in zip(range(i+1, N), scores):
                pair_cache.add(i, j, score)

        return pair_cache.get_list()

    def generate_score_pairs(self, text_list:list[str]):
        self._fit(text_list)
        sparse_vector = self._sparse_vectorize(text_list)
        signatures = self._build_signatures(sparse_vector)

        return self._create_pairs(signatures)

