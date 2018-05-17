import tensorflow as tf


class Seq2Seq:
    batch_size=0
    sequence_length=0
    answers_words_2_ints={}
    questions_words_2_ints={}
    encoding_embedding_size=0
    decoding_embedding_size=0
    rnn_size=0
    num_layers=0
    learning_rate=0

    def __init__(
      self,
      batch_size=64,
      sequence_length=25,
      answers_words_2_ints={},
      questions_words_2_ints={},
      encoding_embedding_size=512,
      decoding_embedding_size=512,
      rnn_size=512,
      # Number of encoder and decoder layers
      num_layers=3,
      learning_rate=0.01,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.answers_words_2_ints = answers_words_2_ints
        self.questions_words_2_ints = questions_words_2_ints
        self.encoding_embedding_size = encoding_embedding_size
        self.decoding_embedding_size = decoding_embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

    def compile(self,
                mode='training'):
        # Initiating graph inputs
        inputs, targets, lr, keep_prob = self.model_inputs()

        # Setting the sequence length
        _sequence_length = tf.placeholder_with_default(self.sequence_length, None, name='sequence_length')

        # Getting the shape of the inputs tensor
        input_shape = tf.shape(inputs)

        # Getting the training and testing predictions
        training_predictions, test_predictions = self._build_graph(
            tf.reverse(inputs, [-1]),  # why reverse inputs? read the seq2seq doc
            targets,
            keep_prob,
            self.batch_size,
            _sequence_length,
            len(self.answers_words_2_ints),
            len(self.questions_words_2_ints),
            self.encoding_embedding_size,
            self.decoding_embedding_size,
            self.rnn_size,
            self.num_layers,
            self.questions_words_2_ints)

        # Optimization
        with tf.name_scope('optimizaiton'):
            # Loss
            loss_error = tf.contrib.seq2seq.sequence_loss(
                training_predictions,
                targets,
                tf.ones([input_shape[0], self.sequence_length]))
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = optimizer.compute_gradients(loss_error)
            clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.),
                                  grad_variable)
                                 for grad_tensor, grad_variable in gradients
                                 if grad_tensor is not None]
            optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

        return training_predictions, test_predictions, optimizer_gradient_clipping, loss_error

    def model_inputs(self):
        ''' Creating placeholders and targets '''
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, targets, learning_rate, keep_prob

    def preprocess_targets(self, targets, word2int, batch_size):
        ''' Preprocessing the targets '''
        # Everything except for the first token
        left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
        # Grab everything except for the last token
        # Answers without the end
        right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        # Horizontal concat
        preprocessed_targets = tf.concat([left_side, right_side], 1)
        return preprocessed_targets

    def encoder_rnn(self,
                    rnn_inputs,
                    rnn_size,
                    num_layers,
                    keep_prob,
                    sequence_length):
        with tf.variable_scope('encoding') as encoding_scope:
            ''' Creating the encoder RNN layer '''
            # with tf.variable_scope('encoding') as encoding_scope:
            # What does BasicLSTMCell do? https://stackoverflow.com/questions/46134806/what-does-basiclstmcell-do
            # Why LSTM layer is not called "layer"? https://github.com/tensorflow/tensorflow/issues/14693
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            # Deactivating certain portions of cells
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(
                lstm, input_keep_prob=keep_prob)

            encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)

            # Reference: https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell,
                cell_bw=encoder_cell,
                sequence_length=sequence_length,
                inputs=rnn_inputs,
                dtype=tf.float32)
            return encoder_state


    # Decoding the training set
    def decode_training_set(self,
                            encoder_state,
                            decoder_cell,
                            decoder_embedded_input,
                            sequence_length,
                            decoding_scope,
                            output_function,
                            keep_prob,
                            batch_size):
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

    def decode_test_set(self,
                        encoder_state,
                        decoder_cell,
                        decoder_embeddings_matrix,
                        sos_id,
                        eos_id,
                        maximum_length,
                        num_words,
                        decoding_scope,
                        output_function,
                        keep_prob,
                        batch_size):
        # TODO: Find out what exactly this part of the code does
        # Decoding the test/validation set
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

    def decoder_rnn(self,
                    decoder_embedded_input,
                    decoder_embedding_matrix,
                    encoder_state,
                    num_words,
                    sequence_length,
                    rnn_size,
                    num_layers,
                    word2int,
                    keep_prob,
                    batch_size):
        # Creating the decoder RNN
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
            # training_predictions = logits
            # What is logits layer?
            # https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow
            training_predictions = self.decode_training_set(
                encoder_state,
                decoder_cell,
                decoder_embedded_input,
                sequence_length,
                decoding_scope,
                output_function,
                keep_prob,
                batch_size)

            decoding_scope.reuse_variables()
            # SOS -> start of sentence
            # EOS -> End of sentence
            test_predictions = self.decode_test_set(
                encoder_state,
                decoder_cell,
                decoder_embedding_matrix,
                word2int['<SOS>'],
                word2int['<EOS>'],
                sequence_length - 1,
                num_words,
                decoding_scope,
                output_function,
                keep_prob, batch_size)
            return training_predictions, test_predictions

    def _build_graph(self, inputs, targets, keep_prob, batch_size, sequence_length,
                      answers_num_words, questions_num_words,
                      encoder_embedding_size, decoder_embedding_size, rnn_size,
                      num_layers, questions_words_2_ints):
        # building seq2seq model
        # Maps a sequence of symbols to a sequence of embeddings
        encoder_embedded_input = tf.contrib.layers.embed_sequence(
            inputs,
            answers_num_words + 1,
            encoder_embedding_size,
            initializer=tf.random_uniform_initializer(0, 1))
        encoder_state = self.encoder_rnn(encoder_embedded_input,
                                    rnn_size, num_layers,
                                    keep_prob, sequence_length)
        preprocessed_targets = self.preprocess_targets(targets,
                                                  questions_words_2_ints,
                                                  batch_size)
        decoder_embeddings_matrix = tf.Variable(
            tf.random_uniform(
                [questions_num_words + 1, decoder_embedding_size], 0, 1
            )
        )
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,
                                                        preprocessed_targets)
        training_predictions, test_predictions = self.decoder_rnn(
            decoder_embedded_input,
            decoder_embeddings_matrix,
            encoder_state,
            questions_num_words,
            sequence_length,
            rnn_size,
            num_layers,
            questions_words_2_ints,
            keep_prob, batch_size)
        return training_predictions, test_predictions


def main():
    from .data import process_count_vectorization
    sorted_clean_questions, sorted_clean_answers, questions_words_2_counts, answers_words_2_counts, answers_counts_2_words = process_count_vectorization()
    # Defining session
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # Getting the training and testing predictions
    model = Seq2Seq(
        batch_size=64,
        sequence_length=25,
        answers_words_2_ints=answers_words_2_counts,
        questions_words_2_ints=questions_words_2_counts,
        encoding_embedding_size=512,
        decoding_embedding_size=512,
        rnn_size=512,
        num_layers=3,
        learning_rate=0.01
    )
    training_predictions, test_predictions, optimizer_gradient_clipping, loss_error = model.compile()
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./output/chatbot-tfboard/2')
    writer.add_graph(session.graph)


if __name__ == "__main__":
    main()

