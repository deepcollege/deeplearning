import tensorflow as tf


class Word2vec:
  batch_size = 20 # num batches, default is set to 20
  embedding_size = 2 # word embedding size
  num_sampled = 15 # Number of negative examples to sample.
  vocab_size = 0 # Vocab size created from the data API
  loss = None # TF Loss
  optimizer = None # Optimizer
  gpu_dynamic_memory_growth = False
  session = None # TF Session

  def __init__(
      self,
      batch_size = 20,
      embedding_size = 2,
      num_sampled = 15,
      vocab_size = 0,
      gpu_dynamic_memory_growth = False
  ):
    # Hyperparams
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.num_sampled = num_sampled
    self.vocab_size = vocab_size
    self.gpu_dynamic_memory_growth = gpu_dynamic_memory_growth

    self.X = tf.placeholder(tf.int32, shape=[batch_size])
    self.y = tf.placeholder(tf.int32, shape=[batch_size, 1])

  def compile(self):
    # Initiate a session
    self.session = self._create_session()

    # Look up embeddings for inputs.
    # The conversion of 10,000 columned matrix into a 200 columned matrix is called word embedding.
    embeddings = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, self.X)  # lookup table

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
      tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    # Compute the average NCE loss for the batch.
    self.loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, self.y, embed, self.num_sampled,
                     self.vocab_size))
    # Using the adam optimizer
    self.optimizer = tf.train.AdamOptimizer(1e-1).minimize(self.loss)

  def train_batch(self, inputs, targets):
    _, loss_val = self.session.run([self.optimizer, self.loss],
                                    feed_dict={self.X: inputs, self.y: targets})

  def _create_session(self):
    """Initialize the TensorFlow session"""
    if self.gpu_dynamic_memory_growth:
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      session = tf.Session(config=config)
    else:
      session = tf.Session()

    return session