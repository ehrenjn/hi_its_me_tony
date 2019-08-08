from .consts import WORD2VEC_LUT_FILE
import json
import zlib
import numpy as np

#CHECK TO SEE IF YOU CAN DO ALL THE COSINE SIMILARITIES AT ONCE WITH NUMPY, THEN CHECK TO SEE IF YOU CAN SORT WITH NUMPY, THEN CHECK TO SEE IF DOING ONE OR BOTH OF THESE THINGS IN NUMPY MAKES to_word FASTER


class Vectorizer:

    def __init__(self, lut_file_path = WORD2VEC_LUT_FILE):
        with open(lut_file_path, 'rb') as lookup:
            data = lookup.read()
        data = zlib.decompress(data)
        self._word_to_vect = json.loads(data)
        self._word_to_vect = {
            word: np.array(vect) 
            for word, vect in self._word_to_vect.items()
        }
    
    def to_vect(self, word):
        return self._word_to_vect.get(word, None)

    def to_word(self, vector, num_results = 1):
        cosine_func = lambda word: cosine_similarity(self._word_to_vect[word], vector)
        results = sorted(
            self._word_to_vect.keys(),
            key = cosine_func,
            reverse = True #reverse because a cos of 1 means high similarity 
        )
        if num_results == 1:
            return results[0]
        return results[:num_results]


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))