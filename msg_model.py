from keras.models import load_model
import numpy
import time
from consts import MAX_MSG_LENGTH, NUM_EXTRA_PARAMETERS, NULL_WORD
from word2vec.consts import WORD2VEC_VEC_SIZE
from message_cleaner import clean_msg


class Message:
    '''wrapper for a message consisting of words, a time posted, and whether Tony is the author'''
    
    def __init__(self, words, time_posted, author_is_tony):
        self.words = words
        self.time_posted = time_posted
        self.author_is_tony = author_is_tony

    @staticmethod
    def from_content(content, time_posted, author_is_tony):
        words = clean_msg(content)
        return Message(words, time_posted, author_is_tony)


def normalize_time_delta(delta):
    '''
    converts seconds since previous message sent to a more meaningful guess of how much I think 
    this message is a response to the previous message
    0 = very likely to be a response, 1 = very unlikey to be a response
    '''
    if (delta == 0):
        return 0
    return (
        numpy.tanh(
            numpy.log2(delta/10000) / 2
        ) + 1
    ) / 2


class TimeSeriesInput:
    '''represents a multi-message input to a predictive model'''
    
    def __init__(self, vectorizer, input_messages):
        self._vectorizer = vectorizer
        self._input_messages = input_messages
        self._remove_non_vectorizable_words(vectorizer, *self._input_messages)


    def make_input(self, response_time = None):

        if response_time is None:
            response_time = time.time() #response time is right now by default
        input_size = WORD2VEC_VEC_SIZE + NUM_EXTRA_PARAMETERS #size of each timestep vector
        author_offset = WORD2VEC_VEC_SIZE #index of each time step that should contain information about author
        time_delta_offset = author_offset + 1 #index of each timestep that contains information about time the message was posted
        input_matrix = numpy.zeros((len(self), input_size))

        time_step = 0
        for msg in self._input_messages:
            normalized_time_delta = normalize_time_delta(response_time - msg.time_posted)
            is_author = int(msg.author_is_tony)

            for word in msg.words + [NULL_WORD]: #tack on end of message word
                time_step_array = input_matrix[time_step]
                word_vect = self._vectorizer.to_vect(word)
                time_step_array[:WORD2VEC_VEC_SIZE] = word_vect
                time_step_array[author_offset] = is_author
                time_step_array[time_delta_offset] = normalized_time_delta
                time_step += 1

        return input_matrix


    def _remove_non_vectorizable_words(self, vectorizer, *messages):
        is_good_word = lambda word: vectorizer.to_vect(word) is not None
        for msg in messages:
            msg.words = list(filter(is_good_word, msg.words))

    
    def __len__(self):
        return sum(
            len(msg.words) + 1 #+1 because we mark the end of messages with a null word
            for msg in self._input_messages
        )
