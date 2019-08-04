import json
from consts import HOT_ENCODINGS_FILE, NUM_INDEXED_WORDS, MSGS_FILE


class Encoder:

    def __init__(self, word_list = []):
        self.to_word = word_list
        self.to_index = {word: index for index, word in enumerate(word_list)}
    
    def add(self, word):
        if word not in self.to_index:
            self.to_word.append(word)
            index = len(self.to_word) - 1
            self.to_index[word] = index
            return True
        return False
    
    def get_word(self, index):
        if index < len(self.to_word) and index >= 0:
            return self.to_word[index]
        return None
    
    def get_index(self, word):
        return self.to_index.get(word, None)

    def save(self, path):
        data = json.dumps(self.to_word)
        with open(path, 'w') as save_file:
            save_file.write(data)

    @staticmethod
    def from_file(path):
        with open(path) as json_file:
            data = json_file.read()
            word_list = json.loads(data)
        return Encoder(word_list)

    def __len__(self):
        return len(self.to_word)



if __name__ == "__main__":

    with open(MSGS_FILE) as msgs_file:
        raw = msgs_file.read()
        messages = json.loads(raw)
    all_words = [word for msg in messages for word in msg['content']]
    word_frequencies = {}
    print(f'total words: {len(all_words)}')

    for word in all_words:
        if word in word_frequencies:
            word_frequencies[word] += 1
        else:
            word_frequencies[word] = 1
    print(f"total unique words: {len(word_frequencies)}")

    top_words = list(word_frequencies.keys())
    top_words.sort(key = lambda word: word_frequencies[word], reverse = True)
    top_words = top_words[:NUM_INDEXED_WORDS]
    print(f"top words: { {word: word_frequencies[word] for word in top_words[:10]} }")
    print(f"bottom words: { {word: word_frequencies[word] for word in top_words[-10:]} }")
    print(f"words accounted for: {round(sum(word_frequencies[word] for word in top_words) / sum(word_frequencies.values()) * 100, 2)}%")

    hot_encoder = Encoder(top_words)
    hot_encoder.save(HOT_ENCODINGS_FILE)