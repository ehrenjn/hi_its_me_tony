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
	ALWAYS TRAINING 150 TIMES PER MESSAGES IS PRETTY INEFFICIENT SINCE MOST MESSAGES END WAY BEFORE THE 150TH CHARACTER, HOW DO I FIX THAT WITHOUT IT FORGETTING HOW TO END MESSAGES?
		seems pretty googlable although it might send you down a rabbithole for a completely different type of rnn or something
	should turn into a pip env so I can run it on other machines easier (dockerizing is going a bit overboard) 
	should have an ExportedModel object or something that you export at the end that you can import in tony spark that just spits out a message given 3 messages
		would be useful so that you dont have to worry about char conversion and stuff when you want to use the model in tony
'''

import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
import itertools

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


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


MAX_MSG_LENGTH = 150

class ProcessedMessage:
	"A more useful representation of a message dict"

	def __init__(self, message, char_conv):
		self.content = numpy.zeros(MAX_MSG_LENGTH)
		for index, char in enumerate(message['content']):
			self.content[index] = char_conv.get_num[char]
		self.timestamp = float(message['created_at'])
		self.channel = message['channel']['id']
		self.author = message['author']['id']

def group_messages_by_channel(messages):
	messages_by_channel = dict()
	for msg in short_messages:
		chan_id = msg.channel
		if chan_id not in messages_by_channel:
			messages_by_channel[chan_id] = []
		messages_by_channel[chan_id].append(msg)
	return messages_by_channel

def preprocess_messages(messages, char_conv):
	"remove long messages, turn all message dicts into ProcessedMessages, group by channel, sort each channel by time posted"
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	processed_messages = (ProcessedMessage(m, char_conv) for m in messages)
	messages_by_channel = group_messages_by_channel(short_messages)
    get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = list(sorted(chan, key = get_timestamp) for chan in messages_by_channel) #sort each messages by time
	return messages_by_channel


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
	def __init__(self, messages_by_channel):
		self._channels = messages_by_channel
	
	def __getitem__(self, index):
		pass

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
	model.add(LSTM(neurons_per_hidden_layer, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer))
	model.add(Dropout(0.2)) 
	model.add(Dense(num_outputs, activation='softmax')) #output probability for each char
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy because softmax
	return model



if __name__ == "__main__":
	messages = get_messages()
	chars = CharConverter(messages)
	messages_by_channel = preprocess_messages(messages, chars)
	batch_generator = BatchGenerator(messages_by_channel)
	model = create_model()
	model.fit_generator() #use a generator because I have way too much data to stuff into an array