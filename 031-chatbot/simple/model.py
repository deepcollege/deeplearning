import tensorflow as tf


class Seq2Seq:
	mode = 'training'
	model_hparams = None
	session = None
	input_shape = None
	# training
	optimizer_gradient_clipping = None
	loss_error = None
	inputs = None
	targets = None
	lr = None
	keep_prob = None
	sequence_length = None  # Static sequence length TODO: refactor it to dynamic

	# inference
	test_predictions = None

	def __init__(
		self,
		model_hparams,
	):
		self.model_hparams = model_hparams

	def compile(self,
							mode='training'):
		# Initiating session
		tf.reset_default_graph()
		self.session = self._create_session()

		self.mode = mode
		# Initiating graph inputs
		self.inputs, self.targets, self.lr, self.keep_prob = self.model_inputs()
		# Setting the sequence length
		self.sequence_length = tf.placeholder_with_default(self.model_hparams['sequence_length'], None,
																											 name='sequence_length')

		# Getting the shape of the inputs tensor
		self.input_shape = tf.shape(self.inputs)

		if self.mode == 'training':
			# Getting the training and testing predictions
			self.optimizer_gradient_clipping, self.loss_error = self._build_graph(
				tf.reverse(self.inputs, [-1]),  # why reverse inputs? read the seq2seq doc
				self.targets,
				self.keep_prob,
				self.model_hparams['batch_size'],
				self.sequence_length,
				self.model_hparams['num_questions_word2count'],
				self.model_hparams['num_answers_word2count'],
				self.model_hparams['encoding_embedding_size'],
				self.model_hparams['decoding_embedding_size'],
				self.model_hparams['rnn_size'],
				self.model_hparams['num_layers'],
				self.model_hparams['get_word2int'])

		elif self.mode == 'testing':
			self.test_predictions = self._build_graph(
				tf.reverse(self.inputs, [-1]),  # why reverse inputs? read the seq2seq doc
				self.targets,
				self.keep_prob,
				self.model_hparams['batch_size'],
				self.sequence_length,
				self.model_hparams['num_questions_word2count'],
				self.model_hparams['num_answers_word2count'],
				self.model_hparams['encoding_embedding_size'],
				self.model_hparams['decoding_embedding_size'],
				self.model_hparams['rnn_size'],
				self.model_hparams['num_layers'],
				self.model_hparams['get_word2int'])
		else:
			raise ValueError('Invalid mode detected!', self.mode)

		# Init before returning
		# TODO: Refactor
		self.session.run(tf.global_variables_initializer())
		self._build_tensorboard()

	def model_inputs(self):
		''' Creating placeholders and targets '''
		inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
		targets = tf.placeholder(tf.int32, [None, None], name='targets')
		learning_rate = tf.placeholder(tf.float32, name='learning_rate')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		return inputs, targets, learning_rate, keep_prob

	def preprocess_targets(self, targets, get_word2int, batch_size):
		''' Preprocessing the targets '''
		# Everything except for the first token
		left_side = tf.fill([batch_size, 1], get_word2int('<SOS>'))
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

	def decode_training_set(self,
													encoder_state,
													decoder_cell,
													decoder_embedded_input,
													sequence_length,
													decoding_scope,
													output_function,
													keep_prob,
													batch_size,
													targets):
		'''
        We need decode_training_set to decode the encoded questions and
        answers of the training set (second part of the Seq2Seq model)
        '''
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

		training_predictions = output_function(decoder_output_dropout)

		# Optimization
		# Loss
		# https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
		with tf.variable_scope('optimizer'):
			loss_error = tf.contrib.seq2seq.sequence_loss(
				training_predictions,
				targets,
				tf.ones([
					self.input_shape[0], self.model_hparams['sequence_length']
				]))
			optimizer = tf.train.AdamOptimizer(self.model_hparams['learning_rate'])
			gradients = optimizer.compute_gradients(loss_error)
			# Gradient clipping
			clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.),
														grad_variable)
													 for grad_tensor, grad_variable in gradients
													 if grad_tensor is not None]
			optimizer_gradient_clipping = optimizer.apply_gradients(
				clipped_gradients)
			return optimizer_gradient_clipping, loss_error

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
		'''
        we need decode_test_set to decode the encoded questions and answers of
        either the validation set or simply new predictions that are not used
         anyway in the training.
        '''
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
									get_word2int,
									keep_prob,
									batch_size,
									decoding_scope,
									targets):
		# Creating the decoder RNN
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
		if self.mode == 'training':
			optimizer_gradient_clipping, loss_error = self.decode_training_set(
				encoder_state,
				decoder_cell,
				decoder_embedded_input,
				sequence_length,
				decoding_scope,
				output_function,
				keep_prob,
				batch_size,
				targets)
			return optimizer_gradient_clipping, loss_error
		elif self.mode == 'testing':
			decoding_scope.reuse_variables()
			# SOS -> start of sentence
			# EOS -> End of sentence
			test_predictions = self.decode_test_set(
				encoder_state,
				decoder_cell,
				decoder_embedding_matrix,
				get_word2int('<SOS>'),
				get_word2int('<EOS>'),
				sequence_length - 1,
				num_words,
				decoding_scope,
				output_function,
				keep_prob, batch_size)
			return test_predictions

	def train_batch(self, inputs, targets, learning_rate):
		_, batch_training_loss_error = self.session.run(
			[self.optimizer_gradient_clipping, self.loss_error], {
				self.inputs: inputs,
				self.targets: targets,
				self.lr: learning_rate,
				self.sequence_length: targets.shape[1],
				self.keep_prob: self.model_hparams['keep_probability']
			})
		return batch_training_loss_error

	def validate_batch(self, inputs, targets, learning_rate):
		batch_validation_loss_error = self.session.run(
			self.loss_error, {
				self.inputs: inputs,
				self.targets: targets,
				self.lr: learning_rate,
				self.sequence_length: targets.shape[1],
				self.keep_prob: 1
			}
		)
		return batch_validation_loss_error

	def save_model(self, checkpoint):
		saver = tf.train.Saver()
		saver.save(self.session, checkpoint)

	def _build_graph(self, inputs, targets, keep_prob, batch_size, sequence_length,
									 answers_num_words, questions_num_words,
									 encoder_embedding_size, decoder_embedding_size, rnn_size,
									 num_layers, get_word2int):
		with tf.variable_scope('seq2seq'):
			with tf.variable_scope('encoding'):
				# building seq2seq model
				# Maps a sequence of symbols to a sequence of embeddings
				encoder_embedded_input = tf.contrib.layers.embed_sequence(
					inputs,
					answers_num_words + 1,
					encoder_embedding_size,
					initializer=tf.random_uniform_initializer(0, 1))
				encoder_state = self.encoder_rnn(
					encoder_embedded_input,
					rnn_size,
					num_layers,
					keep_prob,
					sequence_length)
			with tf.variable_scope('decoding') as decoding_scope:
				preprocessed_targets = self.preprocess_targets(targets,
																											 get_word2int,
																											 batch_size)
				decoder_embeddings_matrix = tf.Variable(
					tf.random_uniform(
						[questions_num_words + 1, decoder_embedding_size], 0, 1
					)
				)
				decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,
																												preprocessed_targets)
				if self.mode == 'training':
					optimizer_gradient_clipping, loss_error = self.decoder_rnn(
						decoder_embedded_input,
						decoder_embeddings_matrix,
						encoder_state,
						questions_num_words,
						sequence_length,
						rnn_size,
						num_layers,
						get_word2int,
						keep_prob,
						batch_size,
						decoding_scope,
						targets)
					return optimizer_gradient_clipping, loss_error
				elif self.mode == 'testing':
					test_predictions = self.decoder_rnn(
						decoder_embedded_input,
						decoder_embeddings_matrix,
						encoder_state,
						questions_num_words,
						sequence_length,
						rnn_size,
						num_layers,
						get_word2int,
						keep_prob,
						batch_size,
						decoding_scope,
						targets)
					return test_predictions
				else:
					raise ValueError('Invalid mode detected!', self.mode)

	def _build_tensorboard(self):
		writer = tf.summary.FileWriter('./output/chatbot-tfboard/2')
		writer.add_graph(self.session.graph)

	def _create_session(self):
		"""Initialize the TensorFlow session"""
		if self.model_hparams['gpu_dynamic_memory_growth']:
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			session = tf.Session(config=config)
		else:
			session = tf.Session()

		return session


def main():
	from .data import Dataset
	ds = Dataset()
	ds.load()
	# Getting the training and testing predictions

	model_hparams = dict({
		# Actual hyperparameters
		'batch_size': 64,
		'sequence_length': 25,
		'encoding_embedding_size': 512,
		'decoding_embedding_size': 512,
		'rnn_size': 512,
		'num_layers': 3,
		'gpu_dynamic_memory_growth': False,
		'keep_probability': 0.5,

		# Static values
		'num_questions_word2count': ds.sub.num_questions_word2count,
		'num_answers_word2count': ds.sub.num_answers_word2count,
		'get_word2int': ds.sub.get_word2int,
	})
	model = Seq2Seq(
		model_hparams=model_hparams
	)
	model.compile()


if __name__ == "__main__":
	main()
