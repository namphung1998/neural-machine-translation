"""
Utilities for building vocabulary
"""

def read_corpus(file_path, source):
    data = []
    with open(file_path, 'r') as corpus:
        for line in corpus.readlines():
            sentence = line.strip().split(' ')

            if source == 'tgt':
                sentence = ['<s>'] + sentence + ['</s>']
            data.append(sentence)
    return data

if __name__ == "__main__":
    data = read_corpus('en_es_data/test.txt', 'tgt')
    print(data)
