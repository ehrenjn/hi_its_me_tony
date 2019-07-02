'''
need to:
	filter out messages longer than 150 chars
		0 fill messages that are < 150 chars
		response should also be (zero filled) 150 chars
	group by channels
	order by time posted
	convert messages to ints
	either:
		don't train on 4 messages in which the 4th is made by the same person, or
		add in an input for each message that's 1 if the message was "your's", and 0 otherwise
			"you" are the person who typed the 4th message
	have another input for each message that tells you how long ago the message was posted
look into:
	TURNS OUT TIME SEQUENCE SIZE CAN ONLY DIFFER BATCH TO BATCH SO I HAVE TO MAKE BATCHES OF MESSAGES THAT HAVE THE SAME RESPONSE LENGTH
		ALSO FIX BATCH GENERATION SO THAT THE LAST BATCHES FOR EACH CHANNEL ARE SMALLER BUT ACTUALLY EXIST
		SHOULD I BE ABLE TO GENERATE A TIME SERIES FROM THE COMBINED MESSAGE OBJECT? OR SHOULD I JUST DO A NESTED FOR LOOP?
			maybe store a numpy array in ProcessedMessage that's the correct size and I just need to edit it a bit each time I gen a time series?
				and then it's better to gen time series from ProcessedMessage because I want to hide that implementation!
	should turn into a pip env so I can run it on other machines easier (dockerizing is going a bit overboard) 
	should have an ExportedModel object or something that you export at the end that you can import in tony spark that just spits out a message given 3 messages
		would be useful so that you dont have to worry about char conversion and stuff when you want to use the model in tony
'''

import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
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
		self.all_chars = {char for msg in messages for char in msg['content']}
		num_chars = len(self.all_chars) #used to normalize index to be between 0 and 1
		self.get_char = {index/num_chars: char for index, char in enumerate(self.all_chars, start = 1)}
		self.get_char[0] = '' #used for padding
		self.get_num = {char: num for num, char in self.get_char}


def normalize_time_delta(delta):
	'''
	converts seconds since previous message sent to a more meaningful guess of how much I think 
	this message is a response to the previous message
	0 = very likely to be a response, 1 = very unlikey to be a response
	'''
	return (
		numpy.tanh(
			numpy.log2(delta/10000) / 2
		) + 1
	) / 2


class TimeSeriesMessage:
	"Represents a message as arrays of input and output time series"

	def __init__(self, response, previous_messages, char_conv):
		content_list = [[char_conv.get_num[char]] for char in response['content']] #each char is in it's own list because time series output must be 2D
		self._time_series_output = numpy.array(content_list) #store in array to use (slightly) less memory
		self._base_input = self._make_base_input(previous_messages, response, char_conv)
	
	def _make_base_input(self, previous_messages, response, char_conv):
		base_input = numpy.zeros(MODEL_INPUT_SIZE - MAX_MSG_LENGTH) #base input is all inputs minus the response content
		response_timestamp = float(response['created_at'])
		response_author = response['author']['id']

		ary_index_offset = 0 #current offset in the base_input array
		offset_per_msg = MAX_MSG_LENGTH + 2 #+2 because theres 2 inputs per message other than the content of the message
		author_offset = MAX_MSG_LENGTH 
		time_offset = MAX_MSG_LENGTH + 1

		for msg in previous_messages:
			for char_num, char in enumerate(msg):
				base_input[ary_index_offset + char_num] = char_conv.get_num(char)
			same_user = float(response_author == msg['author']['id']) #whether the same user wrote msg and the response
			base_input[ary_index_offset + author_offset] = same_user
			time_delta = response_timestamp - float(msg['created_at']) #time between msg and response
			base_input[ary_index_offset + time_offset] = normalize_time_delta(time_delta)
			ary_index_offset += offset_per_msg
		return base_input

	def __len__(self):
		return len(self._time_series_output)
	
	def fill_time_series(self, time_series_input, time_series_output):
		"fills 2 2D numpy arrays with time series input and output"
		response_input_offset = MODEL_INPUT_SIZE - MAX_MSG_LENGTH #how far into time_series_input the response begins
		for time_step in range(len(self)): #generating time series instead of saving it in the class because holding too many time series at once could use too much memory
			time_series_input[time_step, :response_input_offset] = self._base_input
			for index in range(time_step):
				time_series_input[index + response_input_offset] = self._time_series_output[index]
		time_series_output[:] = self._time_series_output #just copy the whole output since we dont need to change anything


def preprocess_messages(messages, char_conv):
	"remove long messages, turn all message dicts into ProcessedMessages"
	
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	messages_by_channel = group_by(short_messages, lambda msg: msg['channel']['id']) #have to group by channel so I can find out which messages go with which responses

	get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = list(sorted(chan, key = get_timestamp) for chan in messages_by_channel) #sort each messages by time
	
	processed_messages = []
	for channel in messages_by_channel:
		responses_only = channel[NUM_INPUT_MSGS:] #responses don't start at 0 because I want to the bot to take some messages as input to predict the next 
		for index, response in enumerate(responses_only, NUM_INPUT_MSGS): 
			msgs = channel[index - 3: index] #last 3 messages before response
			processed_messages.append(TimeSeriesMessage(response, msgs, char_conv))

	return processed_messages


class BatchGenerator(Sequence):
	def __init__(self, time_series_messages, batch_size):
		self._message_batches = self._group_messages_by_batch(time_series_messages, batch_size)
		self._batch_size = batch_size

	def _group_messages_by_batch(self, time_series_messages, batch_size):
		"group messages by batch so that I can implement __getitem__ and __len__ more efficiently/easily"
		grouped_messages = group_by(time_series_messages, len) #group by length of message response since time series length must be the same from batch to batch
		all_batches = []
		for group in grouped_messages.values():
			batch = [group[i: i + batch_size] for i in range(0, len(group), batch_size)] #split each group into batch sized smaller groups
			all_batches.extend(batch)
		return all_batches
	
	def __getitem__(self, index):
		#run .time_series() on every member of self._message_batches[index] and then somehow get it in a 3D array
		batch = self._message_batches[index]
		response_lengths = len(batch[0])  #len(batch[n]) is always the same size since we've grouped by response length
		input_tensor = numpy.empty((len(batch), response_lengths,  MODEL_INPUT_SIZE))
		output_tensor = numpy.empty((len(batch), response_lengths, 1)) #only a single output
		for msg_num, msg in enumerate(batch): #fill tensors with time series data
			msg.fill_time_series(input_tensor[msg_num], output_tensor[msg_num])
		return input_tensor, output_tensor

	def __len__(self):
		return len(self._message_batches)


def create_model(num_inputs, neurons_per_hidden_layer):
	"basically copied and pasted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/"
	model = Sequential()
	model.add(LSTM(neurons_per_hidden_layer, input_shape=(None, num_inputs), return_sequences=True))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer))
	model.add(Dropout(0.2)) 
	model.add(TimeDistributed( #so model outputs sequences instead of just individual chars
		Dense(1, activation='softmax') #output probability for each char
	))
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy because softmax
	return model



if __name__ == "__main__":
	messages = get_messages()
	chars = CharConverter(messages)
	time_series_messages = preprocess_messages(messages, chars)
	batch_generator = BatchGenerator(time_series_messages, BATCH_SIZE)
	model = create_model(MODEL_INPUT_SIZE, NEURONS_PER_HIDDEN_LAYER)
	model.fit_generator(batch_generator) #use a generator because I have way too much data to stuff into an array