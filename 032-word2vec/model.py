import tensorflow as tf

class Word2vec:
  batch_size = 20
  embedding_size = 2
  num_sampled = 15 # Number of negative examples to sample.
  vocab_size = 0

  def __init__(
      self,
      batch_size = 20,
      embedding_size = 2,
      num_sampled = 15,
      vocab_size = 0
  ):
    # Hyperparams
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.num_sampled = num_sampled
    self.vocab_size = vocab_size

    self.X = tf.placeholder(tf.int32, shape=[batch_size])
    self.y = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Look up embeddings for inputs.
    # The conversion of 10,000 columned matrix into a 200 columned matrix is called word embedding.
    embeddings = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, self.X)  # lookup table

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

    # Compute the average NCE loss for the batch.
    loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, self.Y, embed, self.num_sampled, self.vocab_size))
    # Using the adam optimizer
    optimizer = tf.train.AdamOptimizer(1e-1).minimize(loss)
