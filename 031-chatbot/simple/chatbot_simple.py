import numpy as np
import tensorflow as tf
from .data import process_cornell


# TF Meta
# TODO: Check tensorflow version
print('You are using:', tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print('You are using CPU version of TF. Chatbot training should only run using GPU mode')
    exit()


sorted_clean_questions, sorted_clean_answers, answers_ints_2_words = process_cornell()

'''
# Creating placeholders and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, learning_rate, keep_prob


# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    # Everything except for the first token
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # Grab everything except for the last token
    # Answers without the end
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    # Horizontal concat
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# Creating the encoder RNN layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # What does BasicLSTMCell do? https://stackoverflow.com/questions/46134806/what-does-basiclstmcell-do
    # Why LSTM layer is not called "layer"? https://github.com/tensorflow/tensorflow/issues/14693
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # Deactivating certain portions of cells
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=keep_prob)

    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)

    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell,
        cell_bw=encoder_cell,
        sequence_length=sequence_length,
        inputs=rnn_inputs,
        dtype=tf.float32)
    return encoder_state


# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,
                        sequence_length, decoding_scope, output_function,
                        keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states,
        attention_option='bahdanau',
        num_units=decoder_cell.output_size)

    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(
        encoder_state[0],
        attention_keys,
        attention_values,
        attention_score_function,
        attention_construct_function,
        name='attn_dec_train')

    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        training_decoder_function,
        decoder_embedded_input,
        sequence_length,
        scope=decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

    return output_function(decoder_output_dropout)


# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix,
                    sos_id, eos_id, maximum_length, num_words, decoding_scope,
                    output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function = (
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option='bahdanau',
            num_units=decoder_cell.output_size))

    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
        output_function,
        encoder_state[0],
        attention_keys,
        attention_values,
        attention_score_function,
        attention_construct_function,
        decoder_embeddings_matrix,
        sos_id,
        eos_id,
        maximum_length,
        num_words,
        name='attn_dec_inf')

    # decoder_final_state and decoder_final_context_state won't be used
    test_predictions, decoder_final_state, decoder_final_context_state = (
        tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_cell, test_decoder_function, scope=decoding_scope))

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
        training_predictions = decode_training_set(
            encoder_state, decoder_cell, decoder_embedded_input,
            sequence_length, decoding_scope, output_function, keep_prob,
            batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(
            encoder_state, decoder_cell, decoder_embedding_matrix,
            word2int['<SOS>'], word2int['<EOS>'], sequence_length - 1,
            num_words, decoding_scope, output_function, keep_prob, batch_size)
        return training_predictions, test_predictions


# building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length,
                  answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size, rnn_size,
                  num_layers, questions_words_2_ints):
    # Maps a sequence of symbols to a sequence of embeddings
    encoder_embedded_input = tf.contrib.layers.embed_sequence(
        inputs,
        answers_num_words + 1,
        encoder_embedding_size,
        initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input,
                                rnn_size, num_layers,
                                keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets,
                                              questions_words_2_ints,
                                              batch_size)
    decoder_embeddings_matrix = tf.Variable(
        tf.random_uniform(
            [questions_num_words + 1, decoder_embedding_size], 0, 1
        )
    )
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,
                                                    preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(
        decoder_embedded_input, decoder_embeddings_matrix, encoder_state,
        questions_num_words, sequence_length, rnn_size, num_layers,
        questions_words_2_ints, keep_prob, batch_size)
    return training_predictions, test_predictions
'''

# Settings the Hyperparams
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and testing predictions
training_predictions, test_predictions = seq2seq_model(
    tf.reverse(inputs, [-1]), # why reverse inputs? read the seq2seq doc
    targets,
    keep_prob,
    batch_size,
    sequence_length,
    len(answers_words_2_ints),
    len(questions_words_2_ints),
    encoding_embedding_size,
    decoding_embedding_size,
    rnn_size,
    num_layers,
    questions_words_2_ints)

# Setting up the Loss Error, the Optimizer and Gradient Clipping; it is to
# avoid exploding vanishing gradient issues

# TODO: It should be part of the model code
with tf.name_scope('optimizaiton'):
    # Loss
    loss_error = tf.contrib.seq2seq.sequence_loss(
        training_predictions,
        targets,
        tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.),
                          grad_variable)
                         for grad_tensor, grad_variable in gradients
                         if grad_tensor is not None]
    #
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


def apply_padding(batch_of_sequences, word2int):
    '''
    Padding the sequence with the <PAD> token
    :param batch_of_sequences:
    :param word2int:
    :return: ['something', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>'] <- depends on max_sequence_length
    '''
    max_sequence_length = max(
        [len(sequence) for sequence in batch_of_sequences])
    return [
        sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence))
        for sequence in batch_of_sequences
    ]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index:start_index + batch_size]
        answers_in_batch = answers[start_index:start_index + batch_size]
        padded_questions_in_batch = np.array(
            apply_padding(questions_in_batch, questions_words_2_ints))
        padded_answers_in_batch = np.array(
            apply_padding(answers_in_batch, answers_words_2_ints))
        yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into testing and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
'''
batch_index_check_training_loss = 100
batch_index_check_validation_loss = (
    len(training_questions) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = '/output/chatbot_weights.ckpt'


session.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('/output/chatbot-tfboard/2')
writer.add_graph(session.graph)

for epoch in range(1, epochs + 1):
    for batch_index, (padded_question_in_batch,
                      padded_answers_in_batch) in enumerate(
                          split_into_batches(training_questions,
                                             training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run(
            [optimizer_gradient_clipping, loss_error], {
                inputs: padded_question_in_batch,
                targets: padded_answers_in_batch,
                lr: learning_rate,
                sequence_length: padded_answers_in_batch.shape[1],
                keep_prob: keep_probability
            })

        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        # At every batch_index_check_training_loss (e.g. 100), we will print the error
        if batch_index % batch_index_check_training_loss == 0:
            # :>3 means 3 figures; :>4 means 4 figures; .3f means float with 3 decimals
            # TODO: batch_time * batch_index_check_training_loss is complaining about it's still a float?
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(
                epoch,
                epochs,
                batch_index,
                len(training_questions) // batch_size,
                total_training_loss_error / batch_index_check_training_loss,
                int(batch_time * batch_index_check_training_loss)))
            # Recompute total training loss error because we are done with 100 batches
            total_training_loss_error = 0

        # At every batch_index_check_validation_loss we reset total_validation_loss_error
        if batch_index % batch_index_check_validation_loss == 0 and batch_size > 0:
            total_validation_loss_error = 0
            starting_time = time.time()

            for batch_index_validation, (padded_question_in_batch,
                              padded_answers_in_batch) in enumerate(
                split_into_batches(validation_questions,
                                   validation_answers, batch_size)):

                # Validation only contains new data that will be used for observations
                # Probability is 1 when we are doing validation
                batch_validation_loss_error = session.run(
                    loss_error, {
                        inputs: padded_question_in_batch,
                        targets: padded_answers_in_batch,
                        lr: learning_rate,
                        sequence_length: padded_answers_in_batch.shape[1],
                        keep_prob: 1
                    })

                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, '
                  'Batch Validation Time: {:d} seconds'
                  .format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay

            # if lr goes below min_learning_rate
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            # Early stopping
            list_validation_loss_error.append(average_validation_loss_error)
            # If average_validation_loss_error is lower than every validation_loss_error_we got
            # do an earlystopping
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry I do not speak better, I need to practice more')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print('My apologies, I cannot speak better anymore. This is the best I can do.')
        break

print('Game Over')
'''

###### Part 4 - TESTING THE SEQ2SEQ

# Loading the weights and Running the session
ouput_checkpoint = './output/chatbot_weights.ckpt'
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, ouput_checkpoint)


# Converting the questions from strings to list of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


while(True):
    question = raw_input('You: ')
    print('The bot is going to reply to ', question)
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questions_words_2_ints)
    question = question + [questions_words_2_ints['<PAD>']] * (25 - len(question))
    # Creating a fake batch because it always wants the batch size of 25
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    # We are interested in the first element
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    # Getting values of each token ID
    for i in np.argmax(predicted_answer, 1):
        if answers_ints_2_words[i] == 'i':
            token = 'I'
        elif answers_ints_2_words[i] == '<EOS>':
            token = '.'
        elif answers_ints_2_words[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answers_ints_2_words[i]
        answer += token
        if token == '.':
            break
    print('Chatbot:', answer)
