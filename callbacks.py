from keras.callbacks import Callback
import requests
import numpy



class EveryNBatches(Callback):
    "callback to run a function every n batches"

    def __init__(self, freq, func):
        self._freq = freq
        self._func = func
        self._epoch = -1

    def on_batch_end(self, batch, logs = {}):
        if batch % self._freq == 0:
            logs['epoch'] = self._epoch
            self._func(self.model, logs)
    
    def on_epoch_begin(self, epoch, logs = {}):
        self._epoch = epoch
    
    def set_func(self, new_func):
        self._func = new_func


def call_every_n_batches(n):
    def decorator(func):
        return EveryNBatches(n, func)
    return decorator


@call_every_n_batches(300)
def save_model(model, logs):
    name = f"{logs}.h5"
    name = name.replace("'", '').replace(":", "=")
    model.save(name)


def remove_trailing_nulls(char_list):
    nulls_begin = 0
    for index, char in enumerate(char_list):
        if char != '':
            nulls_begin = index + 1
    return char_list[:nulls_begin]


class Predictor:

    def __init__(self, BatchGenerator, preprocess_messages, char_conv):
        self._char_conv = char_conv
        messages = Predictor.make_sample_messages()
        messages = preprocess_messages(messages, char_conv)
        generator = BatchGenerator(messages, 1, char_conv.num_chars + 2, (150, char_conv.num_chars))
        self._batch = generator[0][0]


    @staticmethod
    def make_sample_messages():

        generic_author = {
            'id': '1',
            'name': '1'
        }

        generic_channel = generic_author

        sample_messages = [
            {
                'content': "wake up tony",
                'author': generic_author,
                'channel': generic_channel,
                'created_at': '1'
            }, 
            {
                'content': "todays your big day",
                'author': generic_author,
                'channel': generic_channel,
                'created_at': '2'
            },
            {
                'content': "do you know why youre here?",
                'author': generic_author,
                'channel': generic_channel,
                'created_at': '3'
            },
            {
                'content': '',
                'author': {'id': '2', 'name': 'tony'},
                'channel': generic_channel,
                'created_at': '10'
            }
        ]

        return sample_messages


    def predict(self, model):
        data = model.predict(self._batch)[0] #get 0th because it returns a batch output
        output = []
        for hot_encoding in data:
            char_int = numpy.argmax(hot_encoding)
            char = self._char_conv.get_char[char_int]
            output.append(char)
        return output


class DiscordCallback(EveryNBatches):

    webhook_file = "webhook_url.txt"


    def __init__(self, BatchGenerator, preprocess_messages, char_conv):
        super().__init__(50, None)
        self._predictor = Predictor(BatchGenerator, preprocess_messages, char_conv)
        with open(DiscordCallback.webhook_file) as hook_file:
            webhook_url = hook_file.read()
        self.set_func(self.create_logger_function(webhook_url))
    

    def create_logger_function(self, webhook_url):
        def log_to_discord(model, logs): #wrapped in extra func because I need this to not be a method
            loss = logs.get('loss')
            epoch = logs.get('epoch')
            batch = logs.get('batch')
            latest_output = self._predictor.predict(model)
            output_str = ''.join(latest_output)
            output_lst = remove_trailing_nulls(latest_output)
            content = {
                'color': 0x00FF00,
                'title': 'Report card for Tony Spark',
                'description': f"**Epoch:** {epoch}\n**Batch:** {batch}\n**Loss:** {loss}\n",
                'fields': [{
                        'name': "Some of Tony's latest work:",
                        'value': f"{output_str}\n{output_lst}"
                    },
                ],
                'footer': {
                    'text': '- The Spark Academy for lil robits -'
                }
            }
            requests.post(webhook_url, json = {'embeds': [content]})
        return log_to_discord