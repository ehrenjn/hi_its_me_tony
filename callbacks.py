from keras.callbacks import Callback
import requests



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



class Predictor:

    def __init__(self, BatchGenerator, preprocess_messages, char_conv):
        self._char_conv = char_conv
        messages = Predictor.make_sample_messages()
        messages = preprocess_messages(messages, char_conv)
        generator = BatchGenerator(messages, 1, char_conv.num_chars + 2, 150)
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
        data = model.predict(self._batch)[0] #get 0th because it returns a batch out output
        data *= (self._char_conv.num_chars - 1)
        output = ''.join(self._char_conv.get_char.get(int(round(num)), '') for num in data)
        return output


class DiscordCallback(EveryNBatches):

    webhook_url = "https://discordapp.com/api/webhooks/598700402623512724/L06aEejZyr5TRdn-XlRUwMHEJuwDKzY5oUsD8HhabMEOytVAzjvOssOiNvL7O2OzpfbL"


    def __init__(self, BatchGenerator, preprocess_messages, char_conv):
        super().__init__(50, None)
        self._predictor = Predictor(BatchGenerator, preprocess_messages, char_conv)
        self.set_func(self.create_logger_function())
    

    def create_logger_function(self):
        def log_to_discord(model, logs): #wrapped in extra func because I need this to not be a method
            loss = logs.get('loss')
            epoch = logs.get('epoch')
            batch = logs.get('batch')
            content = {
                'color': 0x00FF00,
                'title': 'Report card for Tony Spark',
                'description': f"**Epoch:** {epoch}\n**Batch:** {batch}\n**Loss:** {loss}\n",
                'fields': [{
                    'name': "Some of Tony's latest work:",
                    'value': self._predictor.predict(model)
                }],
                'footer': {
                    'text': '- The Spark Academy for lil robits -'
                }
            }
            requests.post(DiscordCallback.webhook_url, json = {'embeds': [content]})
        return log_to_discord