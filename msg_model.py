#SO WHEN I MAKE THE DISCORD API FOR THIS IT WOULD BE NICE IF COULD SUBCLASS BRAIN INSTEAD OF MAKING A WHOLE NEW CLASS TO JUST HAVE A SINGLE PREDICT FUNCTION AND HAVE DISCORD MESSAGES AUTOMATICALLY CONVERTED TO MessageS
    #but whatever I don't need it 100%
#Brain.predict_message is almost certainly fricked because it assumes nothing is wrapped in extra layers of arrays
#HAVING TO CHECK IF response is None IN TimeSeriesMessage.make_input IS KIND OF A MESS AND HAVING response DEFAULT TO None IS KIND OF BAD TOO
    #would it be possible to have a TimeSeriesInput that another class in ai.py could inherit from?
        #this base class wouldn't have a concept of a response; the make_input function would take a response_time parameter
            #BUT THEN RESPONSE TIME WOULD EITHER HAVE A None DEFAULT OR Brain AND callbacks.py WOULD HAVE TO PROVIDE THE CURRENT time.time()
                #whatever, that's fine

#~~~WHERE THE FRICK DO I CALL clean_msg?
    #~~~Brain.predict_message MAKES THE MOST SENSE BUT IT HAS TO TAKE WHOLE MessageS AS INPUT BECAUSE IT NEEDS TO WORK WITH DISCORD AND THE CALLBACK
    #~~~HMMM RIGHT NOW IM KINDA FEELING THAT THE BEST SOLUTION MIGHT BE FOR THE CALLBACK AND TONY TO DO IT THEMSELVES BUT ITS A BIT GARBAGE BECAUSE ITS JUST A RANDOM API CALL FOR SEEMINGLY NO REASON
        #~~~THE REASON IT FEELS LIKE IT MAKES SENSE IS BECAUSE THE API IM MAKING HERE OPERATES ON MessageS AND ITS UP TO THE USERS TO FIGURE OUT HOW TO MAKE THEM
        #~~~BUT HOLD ON: EVERYTHING THAT USES THIS API IS GONNA HAVE A content (except for when ai.py uses Message)... 
            #~~~SO ACTUALLY NOW THE BEST BET SEEMS TO BE TO HAVE A WAY TO CREATE A Message FROM EITHER words OR content
                #~~~pretty clean because its the python equivilant of having 2 constructors which is basically what I want
                #~~~STILL HAVE TO CREATE MessageS MYSELF... but now I don't have to call some weird outside function first
#~~~GET RID OF Brain.change_model PROBABLY (DONT THINK THE CALLBACK WILL ACTUALLY USE IT AFTERALL)
#~~~HOW AM I GONNA KEEP TRACK OF CHANNELS AGAIN? honestly I think I only need to do that in the batch generator so I dont need any channel bloat in TimeSeriesMessage
    #~~~oh yeah and I think this happens before I would even convert to Messages so... epic
#~~~kind of nasty that Message requires an author_is_tony when we shouldnt really know that until the messages are time series
    #~~~what if we just passed an author to Message and let TimeSeriesMessage handle it?
        #~~~WOULDNT WORK when response param of TimeSeriesMessage is None
    #~~~frick it, I'll just keep it as is right now because it aint super nasty
#~~~STILL DUNNO IF HAVING TO GIVE A VECTORIZER TO EVERY MESSAGE ACTUALLY MAKES SENSE, IT'S STARTING TO SEEM WEIRD 
    #~~~but this stuff should be fairly easy to patch later, just make the word_vects function take a vectorizer parameter I guess
        #~~~or just do that soft vectorizing in TimeSeriesMessage itself because honestly it makes the most sense there
    #~~~IT LOOKS LIKE TO MAKE A BRAIN AND MAKE MESSAGES ID ALWAYS NEED 2 VECTORIZERS FLOATING AROUND WHICH IS PRETTY NASTY SO PROBABLY SHOULD TAKE IT OUT OF Message
        #~~~just wait until you finish the generator in ai.py I guess

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


class TimeSeriesMessage:
    
    def __init__(self, vectorizer, input_messages, response = None):
        self._vectorizer = vectorizer
        self._input_messages = input_messages
        self._output_message = response
        self._remove_non_vectorizable_words(vectorizer, *self._input_messages)
        if self._output_message is not None:
            self._remove_non_vectorizable_words(vectorizer, self._output_message)


    def make_input(self, response_time = None):

        if response_time is None:
            response_time = time.time() #response time is right now by default
        num_timesteps = sum(len(mess.content) + 1 for mess in self._input_messages) #+1 because we mark the end of messages
        input_size = WORD2VEC_VEC_SIZE + NUM_EXTRA_PARAMETERS #size of each timestep vector
        author_offset = WORD2VEC_VEC_SIZE #index of each time step that should contain information about author
        time_delta_offset = author_offset + 1 #index of each timestep that contains information about time the message was posted
        input_matrix = numpy.zeros((num_timesteps, input_size))

        time_step = 0
        for msg in self._input_messages:
            normalized_time_delta = normalize_time_delta(response_time - msg.time_posted)
            is_author = int(msg.author_is_tony)

            for word in msg.content + [NULL_WORD]: #tack on end of message word
                time_step_array = input_matrix[time_step]
                word_vect = self._vectorizer.to_vect(word)
                time_step_array[:WORD2VEC_VEC_SIZE] = word_vect
                time_step_array[author_offset] = is_author
                time_step_array[time_delta_offset] = normalized_time_delta
                time_step += 1
            time_step += 1

        return input_matrix

    
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

    def _remove_non_vectorizable_words(self, vectorizer, *messages):
        for msg in messages:
            msg.words = list(filter(vectorizer.to_vect, msg.words))
    
    def __len__(self):
        return sum(
            len(msg.words) + NUM_EXTRA_PARAMETERS
            for msg in self._input_messages
        )