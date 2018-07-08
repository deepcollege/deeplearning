import tensorflow as tf

class Word2vec:
  batch_size = 20
  embedding_size = 2
  num_sampled = 15 # Number of negative examples to sample.

  def __init__(
      self,
      batch_size = 20,
      embedding_size = 2,
      num_sampled = 15,
  ):
    # Hyperparams
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.num_sampled = num_sampled

    self.X = tf.placeholder(tf.int32, shape=[batch_size])
    self.y = tf.placeholder(tf.int32, shape=[batch_size, 1])
