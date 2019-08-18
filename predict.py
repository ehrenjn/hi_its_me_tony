from keras.models import load_model
import numpy
import time
from word2vec.encoder import Vectorizer


class Brain:

    num_extra_inputs = 2 #author and time since posted


    def __init__(self, model, char_conv):
        self._model = model
        self._char_conv = char_conv


    def predict_message(self, messages):
        input = self._encode_input(messages)
        output = self._model.predict(input)
        return self._decode_output(output)

    
    def _encode_input(self, messages):
        num_timesteps = sum(len(mess.content) + 1 for mess in messages) #+1 because we mark the end of messages
        input_size = self._char_conv.num_chars + Brain.num_extra_inputs
        input_tensor = numpy.zeros((1, num_timesteps, input_size))

        current_time = time.time()
        time_step = 0
        for msg in messages:
            normalized_time_delta = normalize_time_delta(current_time - msg.time_posted)
            is_author = int(msg.author_is_tony)
            for char in msg.content:
                one_hot_index = self._char_conv.get_index[char]
                self._fill_time_step(input_tensor[0, time_step], 
                    one_hot_index, is_author, normalized_time_delta)
                time_step += 1
            self._fill_time_step(input_tensor[0, time_step], 
                0, is_author, normalized_time_delta)
            time_step += 1

        return input_tensor


    def _fill_time_step(self, time_step_array, one_hot_index, is_author, normalized_time_delta):
        author_offset = self._char_conv.num_chars
        time_delta_offset = author_offset + 1
        time_step_array[one_hot_index] = 1
        time_step_array[author_offset] = is_author
        time_step_array[time_delta_offset] = normalized_time_delta
            
    
    def _decode_output(self, model_output):
        data = model_output[0] #get 0th because it returns a batch output
        output = ''
        lst_output = []
        for hot_encoding in data:
            char_int = numpy.argmax(hot_encoding)
            output += self._char_conv.get_char[char_int]
            lst_output.append(self._char_conv.get_char[char_int])
        print(lst_output)
        return output


def brain_from_files(model_path, char_conv_path):
    with open(char_conv_path, 'rb') as char_conv_file:
        char_conv = pickle.load(char_conv_file)
    model = load_model(model_path)
    return Brain(model, char_conv)



if __name__ == "__main__":
    model = brain_from_files(
        "decent models\\ihate.h5", 
        "decent models\\spaceless.pickle"
    )
    while 1:
        messages = []
        
        #t = 1
        for i in range(3):
            content = input('> ')
            msg = Message(content, time.time(), False)
            #msg = Message(content, t, False)
            #t += 1
            messages.append(msg)
        print(model.predict_message(messages))






































class MessageCharacter:
    "A wrapper for a single character in a message"

    __slots__ = ('char', 'time', 'author') #save some memory why don't we

    def __init__(self, char_index, normalized_time_delta, author):
        self.char = char_index
        self.time = normalized_time_delta
        self.author = author


class TimeSeriesMessage:
    "Represents a message as arrays of input and output time series"

    def __init__(self, response, previous_messages, char_conv):
        self._output = self._make_output(response, char_conv)
        self._base_input = self._make_base_input(previous_messages, response, char_conv)
        self._char_conv = char_conv
    
    def _make_output(self, response, char_conv):
        output = numpy.zeros(MAX_MSG_LENGTH)
        for index, char in enumerate(response['content']):
            output[index] = char_conv.get_index[char]
        return output

    def _make_base_input(self, previous_messages, response, char_conv):
        "creates a base input which will be expanded later into the complete input when fill_time_series is called"

        response_timestamp = response['created_at']
        response_author = response['author']['id']
        base_input = []

        for msg in previous_messages:
            same_user = float(response_author == msg['author']['id']) #whether the same user wrote msg and the response
            time_delta = response_timestamp - msg['created_at'] #time between msg and response
            normal_time_delta = normalize_time_delta(time_delta)
            for char in msg['content']:
                char_index = char_conv.get_index[char]
                new_message_char = MessageCharacter(char_index, normal_time_delta, same_user)
                base_input.append(new_message_char)
            message_terminator = MessageCharacter(0, normal_time_delta, same_user)
            base_input.append(message_terminator) #add a 0 at the end of every message to signify the end of the message
        return base_input

    def __len__(self):
        "the number of timesteps in this TimeSeriesMessage"
        return len(self._base_input)
    
    def fill_time_series(self, time_series_input, time_series_output):
        '''
        fills 2 numpy arrays with time series input and output
        time_series_input and time_series_output must be 2D and all 0s
        '''

        author_offset = self._char_conv.num_chars
        time_delta_offset = author_offset + 1
        for time_step, char in enumerate(self._base_input):
            time_series_input[time_step, char.char] = 1
            time_series_input[time_step, author_offset] = char.author
            time_series_input[time_step, time_delta_offset] = char.time
        
        for char_num, hot_index in enumerate(self._output):
            time_series_output[char_num, int(hot_index)] = 1
















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