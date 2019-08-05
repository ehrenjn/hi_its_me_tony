from os import path


WORD2VEC_PATH = path.dirname(__file__)

def abspath(path_end):
    return path.join(WORD2VEC_PATH, path_end)


HOT_ENCODINGS_FILE = abspath("word_one_hot_encoder.json")

WORD2VEC_LUT_FILE = abspath("word2vec.zlib")

NUM_INDEXED_WORDS = 15000

MSGS_FILE = '../parsed_all_messages.json'

WORD2VEC_VEC_SIZE = 300

WINDOW_SIZE = 3

BATCH_SIZE = 1