import json
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.utils import Sequence
import callbacks
from word2vec.utils import group_by
from word2vec.encoder import Vectorizer
from word2vec.consts import WORD2VEC_VEC_SIZE
from msg_model import Message, TimeSeriesInput
from consts import (
	MAX_MSG_LENGTH, NUM_EXTRA_PARAMETERS, NUM_INPUT_MSGS, 
	BATCH_SIZE, NEURONS_PER_HIDDEN_LAYER
)

#technique/some copy & pasting from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


def get_messages():
	with open('parsed_all_messages.json', 'r') as parsed_msg_file:
		raw = parsed_msg_file.read()
	return json.loads(raw)


class TimeSeriesMessage(TimeSeriesInput):
    '''A TimeSeriesInput expanded to also include an output'''

    def __init__(self, vectorizer, input_messages, response):
        super().__init__(vectorizer, input_messages)
        self._output_message = response
        self._remove_non_vectorizable_words(vectorizer, response)

    def make_input(self): 
        response_time = self._output_message.time_posted
        return super().make_input(response_time)

    def make_output(self):
        output = numpy.empty((MAX_MSG_LENGTH, WORD2VEC_VEC_SIZE))
        null_vector = self._vectorizer.to_vect(NULL_WORD)
        for index in range(len(output)):
            if index < len(self._output_message):
                vect = self._vectorizer.to_vect(self._output_message[index])
                output[index] = vect
            else:
                output[index] = null_vector
        return output


class BatchGenerator(Sequence):

	def __init__(self, vectorizer, messages, batch_size, input_size, output_shape):
		time_series_messages = self._get_time_series_messages(messages, vectorizer)
		self._message_batches = self._group_messages_by_batch(time_series_messages, batch_size)
		self._batch_size = batch_size
		self._input_size = input_size
		self._output_shape = output_shape


	def _get_time_series_messages(self, messages, vectorizer):
		"""remove long messages, turn all message dicts into ProcessedMessages"""
		
		short_messages = (m for m in messages if len(m['content']) < MAX_MSG_LENGTH) #dont want massive messages to limit dimensionality 
		messages_by_channel = group_by(short_messages, lambda msg: msg['channel']['id']) #have to group by channel so I can find out which messages go with which responses
	
		get_timestamp = lambda msg: msg['created_at']
		messages_by_channel = list(
			sorted(chan, key = get_timestamp) 
			for chan in messages_by_channel.values() #sort each messages by time
		)
		
		processed_messages = []
		for channel in messages_by_channel:
			responses_only = channel[NUM_INPUT_MSGS:] #responses don't start at 0 because I want to the bot to take some messages as input to predict the next 
			for index, response in enumerate(responses_only, NUM_INPUT_MSGS): 
				msgs = channel[index - 3: index] #last 3 messages before response
				time_series = self._to_time_series(msgs, response, vectorizer)
				processed_messages.append(time_series)
	
		return processed_messages

	
	def _to_time_series(self, msgs, response, vectorizer):
		msgs = [
			Message(
				m['content'], m['created_at'], 
				m['author']['id'] == response['author']['id']
			)
			for m in msgs
		]
		response = Message(response['content'], response['created_at'], True)
		return TimeSeriesMessage(vectorizer, msgs, response)


	def _group_messages_by_batch(self, time_series_messages, batch_size):
		"""group messages by batch so that I can implement __getitem__ and __len__ more efficiently/easily"""
		grouped_messages = group_by(time_series_messages, len) #group by length of message input since time series length must be the same from batch to batch
		all_batches = []
		for group in grouped_messages.values():
			batches = [ #split each group into batch sized smaller groups
				group[i: i + batch_size] 
				for i in range(0, len(group), batch_size)
			] 
			all_batches.extend(batches)
		return all_batches

	
	def __getitem__(self, index):
		batch = self._message_batches[index]
		num_timesteps = len(batch[0]) #number of timesteps are consistent across batch so I can just check the first one
		input_tensor = numpy.zeros((len(batch), num_timesteps, self._input_size))
		output_tensor = numpy.zeros((len(batch), *self._output_shape))
		for msg_num, msg in enumerate(batch): #fill arrays with time series data
			input_tensor[msg_num] = msg.make_input()
			output_tensor[msg_num] = msg.make_output()
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
	model.add(RepeatVector(num_output_timesteps)) #duplicate encoded state (once for each word of output)
	model.add(Dropout(0.2)) 
	model.add(LSTM( #the decoder
		neurons_per_hidden_layer,
		return_sequences = True #this time we're returning every timestep because each timestep of the output is going to be another word
	))
	model.add(Dropout(0.2))
	model.add(TimeDistributed( #LSTM outputs multiple timesteps and we want one Dense layer per timestep
		Dense(
			outputs_per_char,
			activation = 'tanh' #tanh because elements of word embeddings can be negative or positive. WORD EMBEDDINGS ACTUALLY CAN SPAN BEYOND -1 AND 1 BUT BECAUSE WE ONLY LOOK AT THE DIRECTION OF VECTORS TO FIND A MATCH I CAN (hopefully) GET AWAY WITH MAKING THE MAGNITUDE OF THESE OUTPUT VECTORS SMALLER
		)
	))
	model.compile(loss = 'cosine_similarity', optimizer = 'adam') #cosine_similarity because thats how we compare word embeddings
	return model



if __name__ == "__main__":
	print("getting messages...")
	messages = get_messages()
	vectorizer = Vectorizer()
	input_size = WORD2VEC_VEC_SIZE + NUM_EXTRA_PARAMETERS #each input is a vectorized word plus the extra paramters (message age and author)
	output_shape = (MAX_MSG_LENGTH, WORD2VEC_VEC_SIZE) #output is a one hot encoded message
	print("creating batch generator and model...")
	batch_generator = BatchGenerator(vectorizer, messages, BATCH_SIZE, input_size, output_shape)
	model = create_model(input_size, NEURONS_PER_HIDDEN_LAYER, output_shape)
	print(model.summary())
	print("fitting model...")
	model.fit_generator( #use a generator because I have way too much data to stuff into an array
		batch_generator, 
		shuffle = True, 
		epochs = 100,
		callbacks = [callbacks.DiscordCallback(), callbacks.save_model]
	) 