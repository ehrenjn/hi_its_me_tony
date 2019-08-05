from onehot import Encoder as OnehotEncoder
from keras.models import load_model
from consts import HOT_ENCODINGS_FILE, WORD2VEC_LUT_FILE
import zlib


class InputEncoder:
    "encodes words as input that can be fed into a word2vec model"

    def __init__(self, onehot_encoder = None):
        if onehot_encoder is None:
            self._onehot_encoder = OnehotEncoder.from_file(HOT_ENCODER_FILE)
        else:
            self._onehot_encoder = onehot_encoder
    
    def encode(self, ):
        #DONT EVEN KNOW WHAT INPUT WORD2VEC TAKES YET YEESH


class Vectorizer:

    def __init__():
        with open(WORD2VEC_LUT_FILE, 'rb') as lookup:
            data = lookup.read()
        data = zlib.decompress(data)
        self._lookup = json.loads(data)
    
    def __getitem__(self, word):
        return self._lookup.get(word, None)

'''
    def __init__(self, onehot_encoder = None, model = None):
        self._input_encoder = InputEncoder(onehot_encoder)
        if model is None:
            self._model = load_model(WORD2VEC_MODEL_FILE)
        else:
            self._model = model

    def vectorize()
'''