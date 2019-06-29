import json
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools

#basically copied and pasted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


def get_messages():
    with open('parsed_all_messages.json', 'r') as parsed_msg_file:
        raw = parsed_msg_file.read()
    return json.loads(raw)


class CharConverter:
	def __init__(self, messages):
		all_chars = {char for msg in messages for char in msg['content']}
		self.get_char = {index: char for index, char in enumerate(all_chars)}
		self.get_int = {char: index for index, char in enumerate(all_chars)}


MAX_MSG_LENGTH = 150

def make_training_data(messages, char_conv):
	'''
	need to:
		do each 
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
	'''
	short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
	messages_by_channel = itertools.groupby(short_messages, lambda m: m['channel']['id']) #group messages by channel ids
    messages_by_channel = (sorted(chan, key = lambda msg: msg['']) for chan in messages_by_channel)
    for msg in enumerate(short_messages, 3): #start at msg 3 becaus I want to the bot to take 3 messages as input to predict the next 


if __name__ == "__main__":
	messages = get_messages()
	chars = CharConverter(messages)
	# create mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# pick a random seed
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	# generate characters
	for i in range(1000):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\nDone.")