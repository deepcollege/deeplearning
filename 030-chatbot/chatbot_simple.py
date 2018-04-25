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

print(clean_questions)

