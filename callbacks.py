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


@call_every_n_batches(1)
def save_model(model, logs):
    name = f"{logs}.h5"
    name = name.replace("'", '').replace(":", "=")
    print(name)
    model.save(name)



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


def predict(model, batch, char_conv):
    data = model.predict(batch)[0] #get 0th because it returns a batch out output
    data *= (char_conv.num_chars - 1)
    output = ''.join(char_conv.get_char.get(int(round(num)), '') for num in data)
    return output


class DiscordCallback(EveryNBatches):

    webhook_url = "https://discordapp.com/api/webhooks/598700402623512724/L06aEejZyr5TRdn-XlRUwMHEJuwDKzY5oUsD8HhabMEOytVAzjvOssOiNvL7O2OzpfbL"

    def __init__(self, BatchGenerator, preprocess_messages, char_conv):
        super().__init__(50, None)
        messages = preprocess_messages(make_sample_messages(), char_conv)
        generator = BatchGenerator(messages, 1, char_conv.num_chars + 2, 150)
        batch = generator[0][0]
        def new_predict(model):
            return predict(model, batch, char_conv)
        DiscordCallback._predictor = new_predict
        self.set_func(DiscordCallback.log_to_discord)
    
    @staticmethod
    def log_to_discord(model, logs):
        loss = logs.get('loss')
        epoch = logs.get('epoch')
        batch = logs.get('batch')
        content = {
            'color': 0x00FF00,
            'title': 'Report card for Tony Spark',
            'description': f"**Epoch:** {epoch}\n**Batch:** {batch}\n**Loss:** {loss}\n",
            'fields': [{
                'name': "Some of Tony's latest work:",
                'value': DiscordCallback._predictor(model)
            }],
            'footer': {
                'text': '- The Spark Academy for lil robits -'
            }
        }
        requests.post(DiscordCallback.webhook_url, json = {'embeds': [content]})