"""
Word2vec for fun
Dataset: game of throne <https://github.com/nihitx/game-of-thrones-/blob/master/gameofthrones.txt>
"""

import os
import re
import io
import collections

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

print(filtered_sentences)
