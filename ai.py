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
	might want to dockerize so I can run it on better computers easily 
'''

import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


def get_messages():
    with open('parsed_all_messages.json', 'r') as parsed_msg_file:
        raw = parsed_msg_file.read()
    return json.loads(raw)


class CharConverter:
	def __init__(self, messages):
		self.all_chars = {char for msg in messages for char in msg['content']}
		num_chars = len(self.all_chars) #used to normalize index to be between 0 and 1
		self.get_char = {index/num_chars: char for index, char in enumerate(self.all_chars, start = 1)}
		self.get_char[0] = '' #used for padding
		self.get_int = {char: index for index, char in self.get_char}


MAX_MSG_LENGTH = 150

def group_messages_by_channel(messages):
	messages_by_channel = dict()
	for msg in short_messages:
		chan_id = msg['channel']['id']
		if chan_id not in messages_by_channel:
			messages_by_channel[chan_id] = []
		messages_by_channel[chan_id].append(msg)
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


def make_training_data(messages, char_conv):
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	messages_by_channel = group_messages_by_channel(short_messages)
    get_timestamp = lambda msg: float(msg['created_at'])
	messages_by_channel = (sorted(chan, key = get_timestamp) for chan in messages_by_channel) #sort each messages by time if they're not already
    for channel in messages_by_channel:
		for index, response in enumerate(channel, 3): #start at msg 3 because I want to the bot to take 3 messages as input to predict the next 
			msgs = channel[index - 3: index] #last 3 messages before response
			response_time = get_timestamp(response)
			response_author = response['author']['id']
			converted_msgs = []
			for m in msgs: #create the input
				content = m['content']
				int_content = (char_conv.char_to_int[char] for char in content)
				padding = (0 for _ in range(MAX_MSG_LENGTH - len(content)))
				time_before_response = normalize_time_delta(response_time - get_timestamp(m))
				voice = int(m['author']['id'] == response_author) #1 if this is sent by the same person as the response, 0 otherwise
				converted_msgs = itertools.chain(converted_msgs, int_content, padding, [time_before_response], [voice])


def create_model(num_inputs, num_outputs, neurons_per_hidden_layer):
	"basically copied and pasted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/"
	model = Sequential()
	model.add(LSTM(neurons_per_hidden_layer, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2)) #try to not memorize the data
	model.add(LSTM(neurons_per_hidden_layer))
	model.add(Dropout(0.2)) 
	model.add(Dense(num_outputs, activation='softmax')) #output one hot encoded probability for a character
	model.compile(loss='categorical_crossentropy', optimizer='adam') #crossentropy because softmax



if __name__ == "__main__":
	messages = get_messages()
	chars = CharConverter(messages)
	model = create_model()
	model.fit_generator() #use a generator because I have way too much data to stuff into an array