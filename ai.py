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
import itertools

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


MAX_MSG_LENGTH = 150
NUM_INPUT_MSGS = 3 
MODEL_INPUT_SIZE = MAX_MSG_LENGTH * (NUM_INPUT_MSGS + 1) + (2 * NUM_INPUT_MSGS) #input is all the chars from the input message plus the response, and 2 extra parameters per input message for the time and the author


def get_messages():
    with open('parsed_all_messages.json', 'r') as parsed_msg_file:
        raw = parsed_msg_file.read()
    return json.loads(raw)


class CharConverter:
	"to convert message characters to a number and back again"

	def __init__(self, messages):
		self.all_chars = {char for msg in messages for char in msg['content']}
		num_chars = len(self.all_chars) #used to normalize index to be between 0 and 1
		self.get_char = {index/num_chars: char for index, char in enumerate(self.all_chars, start = 1)}
		self.get_char[0] = '' #used for padding
		self.get_num = {char: num for num, char in self.get_char}


class TimeSeriesMessage:
	"Represents a message as arrays of input and output time series"

	def __init__(self, response, previous_messages, char_conv):
		content_list = [[char_conv.get_num[char]] for char in message['content']] #each char is in it's own list because time series output must be 2D
		self._time_series_output = numpy.array(content_list) #store in array to use (slightly) less memory
		self._timestamp = float(message['created_at'])
		self._author = message['author']['id']
	
	def _make_base_input(self):
		base_input = numpy.empty(MODEL_INPUT_SIZE)

	def __len__(self):
		return len(self._time_series_output)
	
	def time_series(self):
		"creates and returns a 2 2D numpy arrays of each time series input and output"
		for SOMETHING #generating time series instead of saving it in the class because holding too many time series at once could use too much memory
		pass


def group_by(items, key_func):
	grouped_items = dict()
	for obj in items:
		key = key_func(obj)
		if key not in grouped_items:
			grouped_items[key] = []
		grouped_items[key].append(obj)
	return grouped_items

'''
def preprocess_messages(messages, char_conv):
	"remove long messages, turn all message dicts into ProcessedMessages, group by channel, sort each channel by time posted"
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	processed_messages = (ProcessedMessage(m, char_conv) for m in messages)
	messages_by_channel = group_messages_by_channel(short_messages)
    get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = list(sorted(chan, key = get_timestamp) for chan in messages_by_channel) #sort each messages by time
	return messages_by_channel
'''

def preprocess_messages(messages, char_conv):
	"remove long messages, turn all message dicts into ProcessedMessages, group by channel, sort each channel by time posted"
	
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	messages_by_channel = group_by(short_messages, lambda msg: msg['channel']['id'])

	get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = list(sorted(chan, key = get_timestamp) for chan in messages_by_channel) #sort each messages by time
	
	processed_messages = []
	for channel in messages_by_channel:
		responses_only = channel[NUM_INPUT_MSGS:] #responses don't start at 0 because I want to the bot to take some messages as input to predict the next 
 		for index, response in enumerate(responses_only, NUM_INPUT_MSGS): 
			msgs = channel[index - 3: index] #last 3 messages before response
			processed_messages.append(TimeSeriesMessage(response, msgs, char_conv))

	return processed_messages

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

class BatchGenerator(Sequence):
	def __init__(self, messages_by_channel, batch_size, output_size):
		self._message_batches = self._group_messages_by_batch(messages_by_channel, batch_size)
		self._batch_size = batch_size

	def _group_messages_by_batch(self, messages_by_channel, batch_size):
		"group messages by batch so that I can implement __getitem__ and __len__ more efficiently/easily"
		message_batches = []
		messages_per_batch = batch_size + NUM_INPUT_MSGS #need 3 additional messages in every batch since we train on a response plus three previous messages
		for channel in messages_by_channel:
			if len(channel) >= messages_per_batch: #ignore channels that are too small for a batch
				for response_num in range(NUM_INPUT_MSGS, len(channel), batch_size): #we'll lose some of the messages at the end of the channel but oh well... the only alternative is to have batches span channels and that gets confusing
					first_msg = response_num - NUM_INPUT_MSGS
					last_msg = response_num + batch_size
					message_batches.append(channel[first_msg: last_msg])
		return message_batches
	
	def __getitem__(self, index):
		messages = self._message_batches[index]
		batch_size = len(messages) - NUM_INPUT_MSGS #not all batches are the same size
		batch_input = numpy.empty((batch_size, FRICK, MODEL_INPUT_SIZE))
		batch_output = #UHHHHHHHhhhhhhHHHHHHHHHHHHHH
		for index, response in enumerate(messages, start = NUM_INPUT_MSGS):
			pass
		return batch_input, batch_output

	def __len__(self):
		return len(self._message_batches)

def make_training_data(messages_by_channel, char_conv):
    for channel in messages_by_channel:
		for index, response in enumerate(channel, 3): #start at msg 3 because I want to the bot to take 3 messages as input to predict the next 
			msgs = channel[index - 3: index] #last 3 messages before response
			converted_msgs = []
			for m in msgs: #create the input
				time_before_response = normalize_time_delta(response.timestamp - m.timestamp)
				voice = int(m.author == response.author) #1 if this is sent by the same person as the response, 0 otherwise
				converted_msgs = itertools.chain(converted_msgs, m.content, padding, [time_before_response], [voice])


def create_model(num_inputs, num_outputs, neurons_per_hidden_layer):
	"basically copied and pasted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/"
	model = Sequential()
	model.add(LSTM(neurons_per_hidden_layer, input_shape=(None, num_inputs), return_sequences=True))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer))
	model.add(Dropout(0.2)) 
	model.add(TimeDistributed( #so model outputs sequences instead of just individual chars
		Dense(num_outputs, activation='softmax') #output probability for each char
	)
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy because softmax
	return model



if __name__ == "__main__":
	messages = get_messages()
	chars = CharConverter(messages)
	messages_by_channel = preprocess_messages(messages, chars)
	batch_generator = BatchGenerator(messages_by_channel)
	model = create_model()
	model.fit_generator() #use a generator because I have way too much data to stuff into an array