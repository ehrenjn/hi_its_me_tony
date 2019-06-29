'''
following: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
should get rid of:
    hyperlinks
        maybe even all messages with links in them?? but then it's hard to find what a message is in response to
        eh, I'm probably gonna train on multiple messages so one missing message isn't too much of a problem
    emojis?
        hmmm, they're practically "words" so they're probably fine
            improper custom emoji useage would be pretty epic tbh
    unicodes?
        dunno if this badboy can handle em
MIGHT WANT TO TRAIN TO JUST SPEAK LIKE A SINGLE USER, MIGHT BE MORE CONSISTENT

long messages: try only messages that are 1000 chars or less maybe? eh, maybe not
@ messages: delete @, save message
unicodes: get rid of em
custom emojis: see what the format is... if it's just :weedbro: then we're good, otherwise frick em
MAKE EVERYTHING LOWERCASE
'''

import json
import re

OLD_GARBAGE_CHARS_REGEX = re.compile(
    r'''[^ !"#$%&'()*+,\-./0123456789:;<=>?@[\\\]^_`abcdefghijklmnopqrstuvwxyz{|}~]'''
)
GARBAGE_CHARS_REGEX = re.compile( #remove most of the chars for lower dimensionality 
    r'''[^ !,.0123456789?abcdefghijklmnopqrstuvwxyz]'''
)

MENTION_REGEX = re.compile(r'<(@&?|#|:\w+?:)\d+?>') #catches people, channels, and custom emojis

URL_REGEX = re.compile(r'https?://\S+?')


def good_content(content):
    content = content.lower()
    content = GARBAGE_CHARS_REGEX.sub('', content) #remove garbage chars
    content = MENTION_REGEX.sub('', content) #remove mentions
    content = URL_REGEX.sub('', content) #remove urls
    return content


with open('all_messages.json', 'r') as messages_file:
    all_messages = json.loads(messages_file.read())

'''
print(len(all_messages))
parsed = (good_content(m['content']) for m in all_messages)
yoinked = [m for m in parsed if len(m) <= 150]
print(len(yoinked))
'''

good_messages = []
for msg in all_messages:
    new_content = good_content(msg['content'])
    if new_content != '':
        msg['content'] = new_content
        good_messages.append(msg)

with open('parsed_all_messages.json', 'w') as output:
    output.write(json.dumps(good_messages))