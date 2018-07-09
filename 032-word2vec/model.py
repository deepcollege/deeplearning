import tensorflow as tf


class Word2vec:
    batch_size = 20    # num batches, default is set to 20
    embedding_size = 2    # word embedding size
    num_sampled = 15    # Number of negative examples to sample.
    vocab_size = 0    # Vocab size created from the data API
    gpu_dynamic_memory_growth = False
    loss = None    # TF Loss
    optimizer = None    # Optimizer
    session = None    # TF Session
    saver = None  # TF Checkpoint saver

    def __init__(self, **kwargs):
        # Hyperparams
        self.batch_size = kwargs.get('batch_size', 20)
        self.embedding_size = kwargs.get('embedding_size', 2)
        self.num_sampled = kwargs.get('num_sampled', 15)
        self.vocab_size = kwargs.get('vocab_size', 0)
        self.gpu_dynamic_memory_growth = kwargs.get('gpu_dynamic_memory_growth', False)

        if self.vocab_size <= 0:
            raise Exception('Cannot create a Word2Vec model with 0 vocab size')

        self.X = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

    def compile(self):
        # Initiate a session
        self.session = self._create_session()

        # Look up embeddings for inputs.
        # The conversion of 10,000 columned matrix into a 200 columned matrix is called word embedding.
        with tf.device("/cpu:0"):
          embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
          embed = tf.nn.embedding_lookup(embeddings, self.X)    # lookup table

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

        # Compute the average NCE loss for the batch.
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, self.y, embed, self.num_sampled, self.vocab_size))
        # Using the adam optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-1).minimize(self.loss)

        # Init the session
        self.session.run(tf.global_variables_initializer())

        # Init TF Saver
        self.saver = tf.train.Saver()

    def train_batch(self, inputs, targets):
        _, loss_val = self.session.run([self.optimizer, self.loss], feed_dict={self.X: inputs, self.y: targets})
        return loss_val

    def save_model(self, checkpoint):
      self.saver.save(self.session, checkpoint)

    def toJSON(self):
        ''' Util func to check current values '''
        return self.__dict__

    def _create_session(self):
        """Initialize the TensorFlow session"""
        if self.gpu_dynamic_memory_growth:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
        else:
            session = tf.Session()

        return session
