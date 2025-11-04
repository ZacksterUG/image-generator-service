from typing import Callable

import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

def cosine_distance_strategy(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine(a, b))

class MessageComparer:
    def __init__(self, model='bert', strategy='cosine'):
        if model == 'bert':
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        else:
            raise ValueError('Unsupported model')

        if strategy == 'cosine':
            self.strategy = cosine_distance_strategy
        elif strategy is Callable:
            self.strategy = strategy
        else:
            raise ValueError('Unsupported strategy')

    def encode(self, string: str) -> np.ndarray:
        return self.model.encode(string)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.strategy(a, b)

def messages_similarity(cmp: MessageComparer, a: str, b: str, eps: float) -> bool:
    enc1 = cmp.encode(a)
    enc2 = cmp.encode(b)

    res = cmp(enc1, enc2)

    return res < eps

