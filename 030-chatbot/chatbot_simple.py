import numpy as np
import tensorflow as tf
import time
import io
import re


lines = io.open('./inputs/movie_lines.txt', encoding='utf8', errors='ignore').read().split('\n')
conversations = io.open('./inputs/movie_conversations.txt', encoding='utf8', errors='ignore').read().split('\n')

id2line = {}

# Step 1: Creating a dict that maps each line to its id
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

'''
Result of Step 1:
Original: L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
Result: { 'L1045': 'They do not!' }
'''

# Step 2: Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    # Example: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    _convo = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", '').replace(' ', '')
    conversations_ids.append(_convo.split(','))

'''
Step 2:
Original: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
Result: Get the lat "['L194', 'L195', 'L196', 'L197']" then replaces [], '' and empty spaces
> Then split the result and create ['L194', 'L195', 'L196', 'L197'] < Python array
'''

# Step 3: Creates questions and answers sequence
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])

'''
Step 3:
Constructs questions and answers using id2line and conversation array
'''


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


# Step 4: cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))


# Step 5: Clean the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


# Step 6: Creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
'''
Step 6:
For example, for a question: can we make this quick  roxanne korrine and andrew barrett are having an incredibly horrendous public break up on the quad  again
It counts each word occurence such as "can" and accumulates the count into word2count dict
'''

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
'''
Step 7: Same as step 6 but against answer
'''

# Step 8: Creating two dictionaries that map the questions words and the answer words
threshold_questions = 20
questions_words_2_ints = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questions_words_2_ints[word] = word_number
        word_number += 1

'''
Step 8:
If word count of a word in word2count is greater than the threshold, add it to
questions_word_2_int
e.g. u'cliff': 2176
'''

# Step 9: Same as step 8 but for answers
threshold_answers = 20
answers_words_2_ints = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answers_words_2_ints[word] = word_number
        word_number += 1

# Step 10: Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questions_words_2_ints[token] = len(questions_words_2_ints) + 1

for token in tokens:
    answers_words_2_ints[token] = len(answers_words_2_ints) + 1

'''
Step 10:
Adding word count (len(questions_words_2_int) + 1) for each token
'''

# Step 11: Creating an inverse dictionary of the answer 2 words 2 int dictionary
answers_ints_2_words = {w_i: w for w, w_i, in answers_words_2_ints.items()}
'''
Step 11:
u'kinda': 2175 -> u'2175': 'kinda'
'''

print(answers_ints_2_words)
exit()

# Adding the EOS to every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and answers into integers
# and replacing all the words that were filtered out by OUT
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_2_ints:
            ints.append(questions_words_2_ints['<OUT>'])
        else:
            ints.append(questions_words_2_ints[word])
    questions_to_int.append(ints)

answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_2_ints:
            ints.append(answers_words_2_ints['<OUT>'])
        else:
            ints.append(answers_words_2_ints[word])
    answers_to_int.append(ints)

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])

