import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.utils import Sequence
import callbacks
import pickle
from word2vec.utils import group_by

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


MAX_MSG_LENGTH = 150
NUM_INPUT_MSGS = 3
NUM_EXTRA_PARAMETERS = 2 #extra paramters are: age of message and message author 

BATCH_SIZE = 64
NEURONS_PER_HIDDEN_LAYER = 256


def get_messages():
	with open('parsed_all_messages.json', 'r') as parsed_msg_file:
		raw = parsed_msg_file.read()
	return json.loads(raw)


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
	
	def save(self, path):
		with open(path, 'wb') as save_file:
			pickle.dump(self, save_file)


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

		response_timestamp = float(response['created_at'])
		response_author = response['author']['id']
		base_input = []

		for msg in previous_messages:
			same_user = float(response_author == msg['author']['id']) #whether the same user wrote msg and the response
			time_delta = response_timestamp - float(msg['created_at']) #time between msg and response
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
	def __init__(self, time_series_messages, batch_size, input_size, output_shape):
		self._message_batches = self._group_messages_by_batch(time_series_messages, batch_size)
		self._batch_size = batch_size
		self._input_size = input_size
		self._output_shape = output_shape

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
		num_timesteps = len(batch[0]) #number of timesteps are consistent across batch so I can just check the first one
		input_tensor = numpy.zeros((len(batch), num_timesteps, self._input_size)) 
		output_tensor = numpy.zeros((len(batch), *self._output_shape))
		for msg_num, msg in enumerate(batch): #fill arrays with time series data
			msg.fill_time_series(input_tensor[msg_num], output_tensor[msg_num])
		return input_tensor, output_tensor

	def __len__(self):
		return len(self._message_batches)


def create_model(num_inputs, neurons_per_hidden_layer, output_shape):
	'''
	encoder/decoder LSTM model based on combination of https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/ 
	and https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
	'''
	num_output_timesteps, outputs_per_char = output_shape
	model = Sequential()
	model.add(LSTM(
		neurons_per_hidden_layer, 
		input_shape = (None, num_inputs), 
		return_sequences = True #return output for every timestep since we're inputting into another LSTM
	))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer)) #outputs encoded state vector
	model.add(RepeatVector(num_output_timesteps)) #duplicate encoded state (once for each letter of output)
	model.add(Dropout(0.2)) 
	model.add(LSTM( #the decoder
		neurons_per_hidden_layer,
		return_sequences = True #this time we're returning every timestep because each timestep of the output is going to be another letter
	))
	model.add(Dropout(0.2))
	model.add(TimeDistributed( #LSTM outputs multiple timesteps and we want one Dense layer per timestep
		Dense(
			outputs_per_char,
			activation = 'softmax' #use softmax because one hot encoding each letter
		)
	))
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy to match softmax
	return model



if __name__ == "__main__":
	print("getting messages...")
	messages = get_messages()
	chars = CharConverter(messages)
	chars.save('char_converter.pickle')
	input_size = chars.num_chars + NUM_EXTRA_PARAMETERS #each input is 1 one hot encoded char plus the extra paramters (message age and author)
	output_shape = (MAX_MSG_LENGTH, chars.num_chars) #output is a one hot encoded message
	print("preprocessing messages...")
	time_series_messages = preprocess_messages(messages, chars)
	print("creating batch generator and model...")
	batch_generator = BatchGenerator(time_series_messages, BATCH_SIZE, input_size, output_shape)
	model = create_model(input_size, NEURONS_PER_HIDDEN_LAYER, output_shape)
	print(model.summary())
	print("fitting model...")
	model.fit_generator( #use a generator because I have way too much data to stuff into an array
		batch_generator, 
		shuffle = True, 
		epochs = 100,
		callbacks = [callbacks.DiscordCallback(BatchGenerator, preprocess_messages, chars), callbacks.save_model]
	) 