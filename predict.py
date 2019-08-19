from keras.models import load_model
from word2vec.encoder import Vectorizer
from consts import FINAL_MODEL_PATH
from msg_model import TimeSeriesInput



class Brain:
    
    def __init__(self, model):
        self._model = model
        self._vectorizer = Vectorizer()

    def predict_message(self, messages):
        time_series = TimeSeriesInput(self._vectorizer, messages)
        input = time_series.make_input()
        input = input.reshape((1, *input.shape)) #have to reshape since predict_on_batch expects input to consist of batches
        output = self._model.predict_on_batch(input)
        output = map(self._vectorizer.to_word, output[0]) #output[0] because predict_on_batch also outputs a batch
        return ' '.join(output)


def load_brain():
    model = load_model(FINAL_MODEL_PATH)
    return Brain(model)