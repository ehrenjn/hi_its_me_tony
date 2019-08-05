#model basically copied and pasted https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
    #changed rmsprop optimizer to adam
    #changed dot product to cosine similarity
    #changed reshaping to not be as weird


from keras import layers, models, callbacks
from keras.preprocessing import sequence
from consts import (
    NUM_INDEXED_WORDS, WORD2VEC_VEC_SIZE, MSGS_FILE, HOT_ENCODINGS_FILE, 
    WINDOW_SIZE, WORD2VEC_LUT_FILE, BATCH_SIZE
)
from onehot import Encoder as OnehotEncoder
import json
from utils import group_by
import numpy as np
import zlib
from keras.utils import Sequence



class Word2VecModel:

    def __init__(self):
        self.model, self.validation_model, self.vectorizer = self._create_models()


    def _create_models(self):
        #model basically copied and pasted https://adventuresinmachinelearning.com/word2vec-keras-tutorial/ with a couple changes

        target_input = layers.Input((1,)) #going to be tossing in 1 word at a time
        context_input = layers.Input((1,))

        #create a layer that takes a 1 hot index as input and outputs an embedding
        embedding = layers.Embedding(
            NUM_INDEXED_WORDS, #one hot encoding dimentionality 
            WORD2VEC_VEC_SIZE, #embedding size
            input_length = 1, #input length is 1 since we're not doing any sequences
        )

        #make a new layer who's input is target_input and output is the embedding layer
        #THIS DOESN'T MAKE 2 DIFFERENT embedding LAYERS, target_input AND context_input ARE EACH ROUTED INTO THE EXACT SAME TENSOR 
        #   (otherwise they would be using different weights and training would be real weird)
        target = embedding(target_input)
        context = embedding(context_input)

        #since an embedding layer expects time series input it returns 2D data that looks like [[1,2,3...]]
        #since we only have one time step we should reshape it into [1,2,3...] to get rid of the useless 2nd dimension
        target = layers.Reshape((WORD2VEC_VEC_SIZE, ))(target)
        context = layers.Reshape((WORD2VEC_VEC_SIZE, ))(context)

        #combine both embeddings
        dot_product = layers.dot( #cosine similarity is just a normalized dot product
            [target, context], 
            axes = -1, #do the dot product over the last axis (our axes are [batch_size, WORD2VEC_VEC_SIZE])
            normalize = True #normalize so that this dot product becomes a cosine similarity
        )

        #put into sigmoid just to scale the cosine properly
        output = layers.Dense(1, activation = 'sigmoid')(dot_product)

        #create and compile model
        model = models.Model(inputs = [target_input, context_input], outputs = output)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam')

        #create 2nd model to run similarity checks on during training
        validation_model = models.Model(inputs = [target_input, context_input], outputs = dot_product)

        #create one final model to convert a word to a vect
        vectorizer = models.Model(inputs = target_input, outputs = target)

        return model, validation_model, vectorizer


def sort_messages(all_msgs):
    channels = group_by(all_msgs, lambda msg: msg['channel']['name'])
    for chan in channels.values():
        chan.sort(key = lambda msg: float(msg['created_at']))
    return [msg for chan in channels.values() for msg in chan]


def make_training_data(all_msgs, hot_encoder):

    def skipgram_input(word, hot_encoder):
        index = hot_encoder.get_index(word)
        return 0 if index is None else index + 1 #the skipgram function uses 0 to mean invalid word

    all_msgs = sort_messages(all_msgs)
    all_content = (msg['content'] for msg in all_msgs)
    all_words = [
        skipgram_input(word, hot_encoder) 
        for message in all_content
        for word in message
    ]

    sampling_table = sequence.make_sampling_table(NUM_INDEXED_WORDS + 1) #make a table that estimates the frequency of each word occuring according to zipf's law
    skip_grams = sequence.skipgrams(
        all_words, NUM_INDEXED_WORDS + 1, window_size = WINDOW_SIZE, sampling_table = sampling_table
    )
    input_pairs, output = skip_grams
    target_input, context_input = map(np.array, zip(*input_pairs)) #reshape input to be in proper form and convert to numpy arrays
    target_input -= 1 #have to convert back from the format that skipgrams wanted
    context_input -= 1

    return [target_input, context_input], output



class Word2VecCallback(callbacks.Callback):

    save_period = 2000 #save every n batches
    validation_period = 500 #check how we're doing every n batchs
    validation_words = ['1', '3', '5', 'five', 'idiot', 'moron']
    num_closest_validators = 10 #find the n closest words to validation_words when validating

    def __init__(self, model, hot_encoder):
        self._model = model
        self._hot_encoder = hot_encoder

    def on_batch_end(self, batch, logs = {}):
        if batch % Word2VecCallback.save_period == 0:
            self._save_model_lut(batch)
        if batch % Word2VecCallback.validation_period == 0:
            self._model_status_check()

    def _save_model_lut(self, batch_num):
        print("saving model...")
        batch = self._make_hot_epoch()
        all_vects = self._model.vectorizer.predict_on_batch(batch)
        word_to_vec = {}
        for word, index in self._hot_encoder.to_index.items():
            serializable_vect = map(float, all_vects[index])
            word_to_vec[word] = list(serializable_vect)
        word_to_vec = json.dumps(word_to_vec)
        word_to_vec = zlib.compress(word_to_vec.encode(), level = 9) #compress cause this is gonna be a big lad
        file_name = WORD2VEC_LUT_FILE.replace('.zlib', f" {batch_num}.zlib")
        with open(file_name, 'wb') as lut:
            lut.write(word_to_vec)
        print("model saved")
        
    def _model_status_check(self):
        print("running validation check...")
        contexts = self._make_hot_epoch()
        for word in Word2VecCallback.validation_words:
            index = self._hot_encoder.get_index(word)
            targets = np.repeat(index, len(contexts))
            similarities = self._model.validation_model.predict_on_batch([targets, contexts])
            similarities = [
                (self._hot_encoder.get_word(index), similarity) 
                for index, similarity in enumerate(similarities)
            ]
            similarities.sort(key = lambda sim: sim[1])
            closest = ', '.join(sim[0] for sim in similarities[:10])
            print(f"closest words to {word}: {closest}")
        print("validation check finished")

    def _make_hot_epoch(self):
        return np.arange(len(self._hot_encoder))


if __name__ == "__main__":

    model = Word2VecModel()
    hot_encoder = OnehotEncoder.from_file(HOT_ENCODINGS_FILE)

    with open(MSGS_FILE) as msgs_file:
        all_msgs = json.loads(msgs_file.read())
    
    training_input, training_output = make_training_data(all_msgs, hot_encoder)
    callback = Word2VecCallback(model, hot_encoder)

    model.model.fit(
        training_input,
        training_output,
        batch_size = BATCH_SIZE,
        epochs = 20,
        callbacks = [callback]
    )