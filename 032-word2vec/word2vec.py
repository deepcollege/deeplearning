"""
Word2vec for fun
Dataset: game of throne <https://github.com/nihitx/game-of-thrones-/blob/master/gameofthrones.txt>
"""

import os
import re
import io
import collections
import numpy as np


def normalise_text(text):
    '''
  Normalises a given text
  :param text:
  :return:
  '''
    text = text.lower()
    # Replacing some known words to parsable words
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r'[^a-z\s]', '', text)

    return text


def chomp(x):
    if x.endswith("\r\n"): return x[:-2]
    if x.endswith("\n") or x.endswith("\r"): return x[:-1]
    return x


file_path = '{input_dir}/inputs/gameofthrones.txt'.format(input_dir=os.getcwd())
# Initial split, still contains the empty lines
lines = io.open(file_path, encoding='utf8', errors='ignore').read().split('\n')

# Normalising each sentence
normalised_sentences = []
for line in lines:
    sentence = normalise_text(line)
    normalised_sentences.append(sentence)

filtered_sentences = []
for line in normalised_sentences:
    if line != u'':
        filtered_sentences.append(line)

# sentences to words and count
words = " ".join(filtered_sentences).split()
count = collections.Counter(words).most_common()
print('Word count', count[:5])

# Build dictionaries
unique_words = [i[0] for i in count]
dic = {w: i for i, w in enumerate(unique_words)}    #dic, word -> id cats:0 dogs:1 ......
voc_size = len(dic)
print('Vocab size:', voc_size)

# Make indexed word data
data = [dic[word] for word in words] #count rank for every word in words
print('Sample data', data[:10], words[:10])

# Let's make a training data for window size 1 for simplicity
window_size = 1
cbow_pairs = []
for i in range(1, len(data) - window_size):
  cbow_pairs.append([[data[i - window_size], data[i + window_size]], data[i]])

print('Context pairs rank ids', cbow_pairs[:5])
print()

cbow_pairs_words = []
for i in range(1, len(words) - window_size):
  cbow_pairs_words.append([[words[i - window_size], words[i + window_size]], words[i]])
print('Context pairs words', cbow_pairs_words[:5])

# Creating the skip-gram
skip_gram_pairs=[]

for c in cbow_pairs:
    skip_gram_pairs.append([c[1],c[0][0]])
    skip_gram_pairs.append([c[1],c[0][1]])
print('skip-gram pairs', skip_gram_pairs[:5])
print()
skip_gram_pairs_words=[]
for c in cbow_pairs_words:
    skip_gram_pairs_words.append([c[1],c[0][0]])
    skip_gram_pairs_words.append([c[1],c[0][1]])
print('skip-gram pairs words', skip_gram_pairs_words[:5])


def get_batch(size):
  assert size < len(skip_gram_pairs)
  X = []
  Y = []
  rdm = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)

  for r in rdm:
    X.append(skip_gram_pairs[r][0])
    Y.append([skip_gram_pairs[r][1]])
  return X, Y


# generate_batch test
print ('Batches (x, y)', get_batch(3))
