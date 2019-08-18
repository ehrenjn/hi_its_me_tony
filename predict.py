from keras.models import load_model
from word2vec.encoder import Vectorizer
from consts import FINAL_MODEL_PATH
from msg_model import TimeSeriesMessage



class Brain:
    
    def __init__(self, model):
        self._model = model
        self._vectorizer = Vectorizer()

    def predict_message(self, messages):
        time_series = TimeSeriesMessage(self._vectorizer, messages)
        input = time_series.make_input()
        output = self._model.predict(input)
        output = map(self._vectorizer.to_word, output)
        return ' '.join(output)


def load_brain():
    model = load_model(FINAL_MODEL_PATH)
    return Brain(model)