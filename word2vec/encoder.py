from onehot import Encoder as OnehotEncoder
from keras.models import load_model
from consts import HOT_ENCODINGS_FILE, WORD2VEC_MODEL_FILE


class InputEncoder:
    "encodes words as input that can be fed into a word2vec model"

    def __init__(self, ):
        self._onehot_encoder = OnehotEncoder.from_file(HOT_ENCODER_FILE)
    
    def encode(self, ):
        #DONT EVEN KNOW WHAT INPUT WORD2VEC TAKES YET YEESH


class Vectorizer:

    def __init__(self):
        self._input_encoder = InputEncoder()
        self._model = load_model(WORD2VEC_MODEL_FILE)

    def vectorize()