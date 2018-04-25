import numpy as np
import tensorflow as tf
import time
import io
import re


lines = io.open('./inputs/movie_lines.txt', encoding='utf8', errors='ignore').read().split('\n')
conversations = io.open('./inputs/movie_conversations.txt', encoding='utf8', errors='ignore').read().split('\n')

id2line = {}

for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    # Example: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    _convo = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", '').replace(' ', '')
    conversations_ids.append(_convo.split(','))

questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])


# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'re", ' are', text)
    text = re.sub(r"\'d", ' would', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", '', text)
    return text


# cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))


clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the naswers words
threshold = 20
questions_words_2_int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questions_words_2_int[word] = word_number
        word_number += 1

answers_words_2_int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answers_words_2_int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries
T_PAD = '<PAD>'
T_EOS = '<EOS>'
T_OUT = '<OUT>'
T_SOS = '<SOS>'
tokens = [T_PAD, T_EOS, T_OUT, T_SOS]
for token in tokens:
    questions_words_2_int[token] = len(questions_words_2_int) + 1

for token in tokens:
    answers_words_2_int[token] = len(answers_words_2_int) + 1

# Creating the inverse dictionary of the answer 2 words 2 int dictionary
answers_words_2_int = {w_i: w for w, w_i, in answers_words_2_int.items()}

# Adding the EOS
for i in range(len(clean_answers)):
    clean_answers[i] += ' {t}'.format(t=T_EOS)

# Translate all the words into associated int value from word 2 int
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_2_int:
            ints.append(questions_words_2_int[T_OUT])
        else:
            ints.append(questions_words_2_int[word])
    questions_to_int.append(ints)
answers_to_int = []

for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_2_int:
            ints.append(answers_words_2_int[T_OUT])
        else:
            ints.append(answers_words_2_int[word])
    answers_to_int.append(ints)


