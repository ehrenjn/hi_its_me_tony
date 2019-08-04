'''
lowecase everything
remove punctuation
no words longer than a certain length (maybe 30 chars?)
should keep @'s (they're like 21 chars long)
maybe just drop the entirety of tony-spark-gaming-official
oh wow I should drop spoiler tags too
almost forgot to get rid of formatting
lordy you have to deal with newlines somehow
SHOULD I DEAL WITH THE REPEATED NEWLINES? seems like its basically my fault that they're there so I should probably fix em
'''

import json
import re

MAX_WORD_LENGTH = 30

PUNCTUATION_REGEX = re.compile(r"""[.,;\(\)'"!?[\]]""") #want to keep ':' for custom emojis

FORMATTING_REGEX = re.compile("[*_|`]")

END_OF_SENTENCE_REGEX = re.compile(r'([^\.])\. ') #don't count '...' as the end of a sentence
#a sentence also has to end with a space because if there isn't a space then it's either there as a meme or at the very end of a message so the period can just be erased

BAD_CHANNELS = [
    "tony-spark-gaming-official"
]

def format_sentence_breaks(content):
    content = content.replace('\n', ' \n ') #space out newlines
    content = END_OF_SENTENCE_REGEX.sub(r'\g<1> \n ', content)
    return content

def valid_word(word, current_words):
    return len(word) <= MAX_WORD_LENGTH \
        and word != '' \
        and not (word == '\n' and (len(current_words) == 0 or current_words[-1] == '\n')) #don't allow multiple newlines in a row or newlines at beginning of message


def clean_msg(content):
    content = content.lower()
    content = format_sentence_breaks(content)
    good_words = []
    for word in content.split(' '):
        word = PUNCTUATION_REGEX.sub('', word)
        word = FORMATTING_REGEX.sub('', word)
        if valid_word(word, good_words):
            good_words.append(word)
    return good_words


if __name__ == "__main__":

    with open('all_messages.json', 'r') as messages_file:
        all_messages = json.loads(messages_file.read())

    good_messages = []
    for msg in all_messages:
        if msg['channel']['name'] not in BAD_CHANNELS:
            new_content = clean_msg(msg['content'])
            if len(new_content) > 0:
                msg['content'] = new_content
                good_messages.append(msg)

    with open('parsed_all_messages.json', 'w') as output:
        output.write(json.dumps(good_messages))