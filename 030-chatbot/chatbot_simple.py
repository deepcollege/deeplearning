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

# Step 12: Adding the EOS to every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

'''
Step 12: On each clean_answer, it append ' <EOS>'
'''

# Step 13: Translating all the questions and answers into integers
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


'''
By looking up questions_wirds_2_ints, loop through clean_questions and try to 
convert all words into integer. If you cannot find a word in questions_words_2_ints
it will just use <OUT> int value
'''

# Step 13: Same as Step 12
answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_2_ints:
            ints.append(answers_words_2_ints['<OUT>'])
        else:
            ints.append(answers_words_2_ints[word])
    answers_to_int.append(ints)

# Step 14: Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


# Creating placeholders and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, learning_rate, keep_prob


# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # Grab everything except for the last token
    # Answers without the end
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    # Horizontal concat
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Creating the encoder RNN layer
def encoder_rnn(
        rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # Deactivating certain portions are deactivated during training
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(
        lstm,
        input_keep_prob=keep_prob)

    encoder_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_dropout] * num_layers)

    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell,
        cell_bw=encoder_cell,
        sequence_length=sequence_length,
        inputs=rnn_inputs,
        dtype=tf.float32)
    return encoder_state


# Decoding the training set
def decode_training_set(
        encoder_state, decoder_cell, decoder_embedded_input,
        sequence_length, decoding_scope, output_function,
        keep_prob, batch_size):
    attention_states = tf.zeroes([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = (
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option='bahdanau',
            num_units=decoder_cell.output_size)
    )

    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(
        encoder_state[0],
        attention_keys,
        attention_values,
        attention_score_function,
        attention_construct_function,
        name='attn_dec_train')

    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell, training_decoder_function, decoder_embedded_input,
        sequence_length, scope=decoding_scope
    )

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)


# Decoding the test/validation set
def decode_test_set(
        encoder_state, decoder_cell, decoder_embedding_matrix,
        sos_id, eos_id, maximum_length, num_words,
        sequence_length, decoding_scope, output_function,
        keep_prob, batch_size):
    attention_states = tf.zeroes([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = (
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option='bahdanau',
            num_units=decoder_cell.output_size)
    )

    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
        output_function,
        encoder_state[0],
        attention_keys,
        attention_values,
        attention_score_function,
        attention_construct_function,
        decoder_embedding_matrix,
        sos_id,
        eos_id,
        maximum_length,
        num_words,
        name='attn_dec_inf')

    # decoder_final_state and decoder_final_context_state won't be used
    test_predictions, decoder_final_state, decoder_final_context_state = (
        tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_cell,
            test_decoder_function,
            scope=decoding_scope
        )
    )

    return test_predictions


# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embedding_matrix,
                encoder_state, num_words, sequence_length, rnn_size,
                num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(
            x,
            num_words,
            None,
            scope=decoding_scope,
            weights_initializer=weights,
            biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embedding_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        return training_predictions, test_predictions


# building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length,
                  answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(
        encoder_embedded_input, rnn_size, num_layers,
        keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(
        targets, questions_words_2_ints, batch_size)
    decoder_embeddings_matrix = tf.Variable(
        tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_words_2_ints,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
