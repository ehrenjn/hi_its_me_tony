from .consts import WORD2VEC_LUT_FILE
import json
import zlib


class Vectorizer:

    def __init__(self, lut_file_path = WORD2VEC_LUT_FILE):
        with open(lut_file_path, 'rb') as lookup:
            data = lookup.read()
        data = zlib.decompress(data)
        self._lookup = json.loads(data)
    
    def __getitem__(self, word):
        return self._lookup.get(word, None)
