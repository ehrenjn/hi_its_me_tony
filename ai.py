'''
MAKE IT SAVE VERSIONS OF THE MODEL PERIODICALLY AS IT TRAINS
LOOK INTO WHAT ELSE YOU CAN PASS TO fit_generator 
TEST SOME OF YOUR FUNCS INDIVIDUALLY (do some sanity checks)
SEE IF YOU CAN MAKE IT SAVE A VERSION OF THE MODEL THAT CAN BE USED WAY MORE EASILY
	(like how I wanted to have an object that I can just paste into tony and not have to worry about any data science stuff)
	might have to just end up making a module that uses the exported net file and then you can call the module

NEED TO ONE HOT ENCODE OUTPUT!!!
	just do get_index instead of get_num to make a base_output and then do some ez stuff in fill_time_series
'''

import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


MAX_MSG_LENGTH = 150
NUM_INPUT_MSGS = 3 
MODEL_INPUT_SIZE = MAX_MSG_LENGTH * (NUM_INPUT_MSGS + 1) + (2 * NUM_INPUT_MSGS) #input is all the chars from the input message plus the response, and 2 extra parameters per input message for the time and the author

BATCH_SIZE = 64
NEURONS_PER_HIDDEN_LAYER = 256


def get_messages():
    with open('parsed_all_messages.json', 'r') as parsed_msg_file:
        raw = parsed_msg_file.read()
    return json.loads(raw)


def group_by(items, key_func):
	"itertools.groupby is scary because it can only be iterated over once and I don't need to be super efficient for just preprocessing"
	grouped_items = dict()
	for obj in items:
		key = key_func(obj)
		if key not in grouped_items:
			grouped_items[key] = []
		grouped_items[key].append(obj)
	return grouped_items


class CharConverter:
	"to convert message characters to a number and back again"

	def __init__(self, messages):
		all_chars = {char for msg in messages for char in msg['content']}
		self.get_index = {char: index for index, char in enumerate(all_chars, start = 1)} #used to one hot encode
		self.get_index[''] = 0 #used for padding
		self.num_chars = len(self.get_index) 
		self.get_char = {index: char for char, index in self.get_index.items()}
		divider = self.num_chars - 1 #to normalize index to be between 0 and 1 (-1 because if theres 10 chars then the max index is only 9 but we want index 9 to become num 1.0)
		self.get_num = {char: index/divider for char, index in self.get_index.items()}


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


class TimeSeriesMessage:
	"Represents a message as arrays of input and output time series"

	def __init__(self, response, previous_messages, char_conv):
		self._base_index_output, self._base_num_output = self._make_base_outputs(response, char_conv)
		self._base_input = self._make_base_input(previous_messages, response, char_conv)
		self._char_conv = char_conv
	
	def _make_base_outputs(self, response, char_conv):
		content_index = [char_conv.get_index[char] for char in response['content']] #use char index because I'm gonna one hot encode the output
		content_index.append(0) #append an extra null on the end so that network can (hopefully) learn where the ends of sentences are
		content_num = [index/char_conv.num_chars for index in content_index]
		return numpy.array(content_index), numpy.array(content_num) #store in arrays to use less memory

	def _make_base_input(self, previous_messages, response, char_conv):
		base_input = numpy.zeros(MODEL_INPUT_SIZE - MAX_MSG_LENGTH) #base input is all inputs minus the response content
		response_timestamp = float(response['created_at'])
		response_author = response['author']['id']

		ary_index_offset = 0 #current offset in the base_input array
		offset_per_msg = MAX_MSG_LENGTH + 2 #+2 because theres 2 inputs per message other than the content of the message
		author_offset = MAX_MSG_LENGTH 
		time_offset = MAX_MSG_LENGTH + 1

		for msg in previous_messages:
			for char_num, char in enumerate(msg['content']):
				base_input[ary_index_offset + char_num] = char_conv.get_num[char]
			same_user = float(response_author == msg['author']['id']) #whether the same user wrote msg and the response
			base_input[ary_index_offset + author_offset] = same_user
			time_delta = response_timestamp - float(msg['created_at']) #time between msg and response
			base_input[ary_index_offset + time_offset] = normalize_time_delta(time_delta)
			ary_index_offset += offset_per_msg
		return base_input

	def __len__(self):
		return len(self._base_index_output)
	
	def fill_time_series(self, time_series_input, time_series_output):
		"fills 2 2D numpy arrays with time series input and output"
		response_input_offset = MODEL_INPUT_SIZE - MAX_MSG_LENGTH #how far into time_series_input the response begins
		for time_step in range(len(self)): #generating time series instead of saving it in the class because holding too many time series at once could use too much memory
			time_series_input[time_step, :response_input_offset] = self._base_input
			for index in range(time_step):
				time_series_input[time_step, index + response_input_offset] = self._base_num_output[index]
			time_series_output[time_step].fill(0)
			time_series_output[time_step, self._base_index_output[time_step]] = 1


def preprocess_messages(messages, char_conv):
	"remove long messages, turn all message dicts into ProcessedMessages"
	
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	messages_by_channel = group_by(short_messages, lambda msg: msg['channel']['id']) #have to group by channel so I can find out which messages go with which responses

	get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = list(
		sorted(chan, key = get_timestamp) for chan in messages_by_channel.values() #sort each messages by time
	)
	
	processed_messages = []
	for channel in messages_by_channel:
		responses_only = channel[NUM_INPUT_MSGS:] #responses don't start at 0 because I want to the bot to take some messages as input to predict the next 
		for index, response in enumerate(responses_only, NUM_INPUT_MSGS): 
			msgs = channel[index - 3: index] #last 3 messages before response
			processed_messages.append(TimeSeriesMessage(response, msgs, char_conv))

	return processed_messages


class BatchGenerator(Sequence):
	def __init__(self, time_series_messages, batch_size, output_size):
		self._message_batches = self._group_messages_by_batch(time_series_messages, batch_size)
		self._batch_size = batch_size
		self._output_size = output_size

	def _group_messages_by_batch(self, time_series_messages, batch_size):
		"group messages by batch so that I can implement __getitem__ and __len__ more efficiently/easily"
		grouped_messages = group_by(time_series_messages, len) #group by length of message response since time series length must be the same from batch to batch
		all_batches = []
		for group in grouped_messages.values():
			batch = [group[i: i + batch_size] for i in range(0, len(group), batch_size)] #split each group into batch sized smaller groups
			all_batches.extend(batch)
		return all_batches
	
	def __getitem__(self, index):
		batch = self._message_batches[index]
		response_lengths = len(batch[0])  #len(batch[n]) is always the same size since we've grouped by response length
		input_tensor = numpy.empty((len(batch), response_lengths, MODEL_INPUT_SIZE))
		output_tensor = numpy.empty((len(batch), response_lengths, self._output_size)) 
		for msg_num, msg in enumerate(batch): #fill tensors with time series data
			msg.fill_time_series(input_tensor[msg_num], output_tensor[msg_num])
		print(output_tensor)
		return input_tensor, output_tensor

	def __len__(self):
		return len(self._message_batches)


def create_model(num_inputs, neurons_per_hidden_layer, num_outputs):
	"basically copied and pasted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/"
	model = Sequential()
	model.add(LSTM(
		neurons_per_hidden_layer, 
		input_shape = (None, num_inputs), 
		return_sequences = True
	))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer))
	model.add(Dropout(0.2)) 
	#model.add(TimeDistributed( #output sequence instead of single char
	model.add(Dense(
		num_outputs,
		activation = 'softmax' #output a probability for each char
	))
	#))
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy because softmax
	return model



if __name__ == "__main__":
	print("getting messages...")
	messages = get_messages()
	chars = CharConverter(messages)
	print("preprocessing messages...")
	time_series_messages = preprocess_messages(messages, chars)
	print("creating batch generator and model...")
	batch_generator = BatchGenerator(time_series_messages, BATCH_SIZE, chars.num_chars)
	model = create_model(MODEL_INPUT_SIZE, NEURONS_PER_HIDDEN_LAYER, chars.num_chars)
	print(model.summary())
	print("fitting model...")
	model.fit_generator(batch_generator, shuffle = True, epochs = 20) #use a generator because I have way too much data to stuff into an array