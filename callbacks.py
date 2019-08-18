from keras.callbacks import Callback
from predict import Brain
from msg_model import Message
import requests
import numpy
from consts import NULL_WORD



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



class DiscordCallback(EveryNBatches):

    webhook_file = "webhook_url.txt"


    def __init__(self):
        super().__init__(50, None)
        with open(DiscordCallback.webhook_file) as hook_file:
            webhook_url = hook_file.read()
        self._webhook_url = webhook_url
        self.set_func(self.log_to_discord)
        self._sample_messages = DiscordCallback._create_sample_messages()
        self._predictor = None


    def log_to_discord(self, model, logs):

        if self._predictor is None:
            self._predictor = Brain(model)

        loss = logs.get('loss')
        epoch = logs.get('epoch')
        batch = logs.get('batch')
        latest_output = self._predictor.predict_message(self._sample_messages)
        output_list = DiscordCallback._remove_trailing_nulls(latest_output.split(' '))

        content = {
            'color': 0x00FF00,
            'title': 'Report card for Tony Spark',
            'description': f"**Epoch:** {epoch}\n**Batch:** {batch}\n**Loss:** {loss}\n",
            'fields': [{
                    'name': "Some of Tony's latest work:",
                    'value': f"{latest_output}\n{output_list}"
                },
            ],
            'footer': {
                'text': '- The Spark Academy for lil robits -'
            }
        }

        requests.post(self._webhook_url, json = {'embeds': [content]})
    

    @staticmethod
    def _create_sample_messages():
        contents = [
            "wake up Tony",
            "today's your big day",
            "do you know why you're here?"
        ]
        messages = [
            Message.from_content(sentence, index, False)
            for index, sentence in enumerate(contents, start = 1) #start counting at 1 because a time of 0 is scary
        ]
        return messages


    @staticmethod
    def _remove_trailing_nulls(word_list):
        nulls_begin = 0
        for index, word in enumerate(word_list):
            if word != NULL_WORD:
                nulls_begin = index + 1
        return word_list[:nulls_begin]