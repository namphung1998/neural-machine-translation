import heapq
from utilities import read_corpus
import json

class Language(object):
    def __init__(self):

        self.word2idx = {}
        self.word2idx['<pad>'] = 0 # padding token
        self.word2idx['<s>'] = 1   # start token
        self.word2idx['</s>'] = 2  # end token
        self.word2idx['<unk>'] = 3 # unknown word

        self.unknown = self.word2idx['<unk>']
        self.idx2word = {val: key for key, val in self.word2idx.items()}

    def __getitem__(self, word):
        # find the index of a word, or the unknown if the word is not known by the language
        return self.word2idx.get(word, self.unknown)

    def __contains__(self, word):
        return word in self.word2idx

    def __len__(self):
        return len(self.word2idx)

    def __repr__(self):
        return 'Language with {} words'.format(len(self))

    def add_word(self, word):
        if word in self:
            return self[word]

        idx = len(self)
        self.word2idx[word] = idx
        self.idx2word[idx] = word
        return idx

    @staticmethod
    def build_from_corpus(corpus, size, frequency_cutoff=2):
        """
        Construct a Language from a given corpus
        @param corpus: (list[str]):     corpus of text returned by read_corpus() function
        @param size: (int):             number of words in the vocabulary
        @param frequency_cutoff (int)   minimum number of occurences to be included
        """
        lang = Language()
        count = {}
        for sentence in corpus:
            for word in sentence:
                if word not in count:
                    count[word] = 1
                else:
                    count[word] += 1
        
        valid_words = [(-num, word) for word, num in count.items() if num >= frequency_cutoff]
        heapq.heapify(valid_words)

        top_k_words = []
        for _ in range(size):
            try:
                _, word = heapq.heappop(valid_words)
                top_k_words.append(word)
            except IndexError:
                break

        for word in top_k_words:
            lang.add_word(word)
        return lang


class Vocab(object):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    @staticmethod
    def build(src_sentences, tgt_sentences, vocab_size, frequency_cutoff):
        assert len(src_sentences) == len(tgt_sentences)

        print('Initializing source vocabulary...')
        src = Language.build_from_corpus(src_sentences, vocab_size, frequency_cutoff)

        print('Initializing target vocabulary...')
        tgt = Language.build_from_corpus(tgt_sentences, vocab_size, frequency_cutoff)

        return Vocab(src, tgt)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump(dict(src_word2idx=self.src.word2idx, tgt_word2Idx=self.tgt.word2idx), json_file, indent=2)

if __name__ == "__main__":
    src_sentences = read_corpus('en_es_data/dev.en', source='src')
    tgt_sentences = read_corpus('en_es_data/dev.es', source='tgt')
    vocab = Vocab.build(src_sentences, tgt_sentences, 100, 2)
    vocab.save('vocab.json')
