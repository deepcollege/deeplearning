import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class DCGAN:
    def __init__(self,
                 img_length,
                 num_colors,
                 d_sizes,
                 g_sizes,
                 keep_prob,
                 learning_rate,
                 n_noise):

        self.img_length = img_length
        self.num_colors = num_colors
        self.latent_dims = g_sizes['n_noise']
        self.momentum = 0.99
        self.n_noise = n_noise
        self.keep_prob = keep_prob

        # For mnist, img_length = 28, num_colors = 1
        self.real_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, img_length, img_length, num_colors],
            name='real_images')

        self.noise = tf.placeholder(
            dtype=tf.float32, shape=[None, self.latent_dims])
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        g = self.build_generator(
            self.noise, keep_prob, self.is_training, g_sizes=g_sizes)
        d_real = self.build_discriminator(
            self.real_images, reuse=None, keep_prob=keep_prob, d_sizes=d_sizes)
        d_fake = self.build_discriminator(
            g, reuse=True, keep_prob=keep_prob, d_sizes=d_sizes)

        vars_g = [var for var in tf.trainable_variables() if 'gen' in var.name]
        vars_d = [var for var in tf.trainable_variables() if 'disc' in var.name]

        # Applying regularizers
        d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
        g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

        # build costs
        self.loss_d_real = self.binary_cross_entropy(tf.ones_like(d_real), d_real)
        self.loss_d_fake = self.binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
        self.loss_g = tf.reduce_mean(self.binary_cross_entropy(tf.ones_like(d_fake), d_fake))
        self.loss_d = tf.reduce_mean(0.5 * (self.loss_d_real + self.loss_d_fake))

        # Optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optimizer_d = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_d + d_reg, var_list=vars_d)
        self.optimizer_g = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_g + g_reg, var_list=vars_g)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def lrelu(self, x):
        return tf.maximum(x, tf.multiply(x, 0.2))

    def binary_cross_entropy(self, x, z):
        eps = 1e-12
        return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

    def build_discriminator(self, X, reuse=None, keep_prob=None, d_sizes=None):
        '''
        d_sizes = {
            'conv_layers': [],
            'dense_layers': []
        }
        '''
        activation = self.lrelu
        with tf.variable_scope('disc', reuse=reuse):
            # Reshaping input
            x = tf.reshape(X, shape=[-1, self.img_length, self.img_length, 1])
            for kernels, filtersz, stride, dropout, apply_batch_norm in d_sizes[
                    'conv_layers']:
                x = tf.layers.conv2d(
                    x,
                    kernel_size=kernels,
                    filters=filtersz,
                    strides=stride,
                    padding='same',
                    activation=activation)
                if dropout:
                    x = tf.layers.dropout(x, keep_prob)

                if apply_batch_norm:
                    x = tf.contrib.layers.batch_norm(
                        x, is_training=self.is_training, decay=self.momentum)
            for units in d_sizes['dense_layers']:
                print('checking dense units ', units)
                x = tf.layers.dense(x, units=units, activation=activation)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
            return x

    def build_generator(self, Z, keep_prob, is_training, g_sizes):
        '''
        # Example
        g_sizes = {
          'd1': 4,
          'd2': 1,
          'dense_layers': [],
          'conv_layers': []
        }
        '''
        activation = self.lrelu
        with tf.variable_scope('gen', reuse=None):
            x = Z
            d1 = g_sizes['d1']
            d2 = g_sizes['d2']
            # Dense layers
            for units, dropout, apply_batch_norm in g_sizes['dense_layers']:
                x = tf.layers.dense(x, units=units, activation=activation)
                if dropout:
                    x = tf.layers.dropout(x, keep_prob)
                if apply_batch_norm:
                    x = tf.contrib.layers.batch_norm(
                        x, is_training=is_training, decay=self.momentum)
                x = tf.reshape(x, shape=[-1, d1, d1, d2])
                x = tf.image.resize_images(x, size=[7, 7])
                # Conv layers
            for kernels, filtersz, stride, dropout, apply_batch_norm in g_sizes[
                    'conv_layers']:
                x = tf.layers.conv2d_transpose(
                    x,
                    kernel_size=kernels,
                    filters=filtersz,
                    strides=stride,
                    padding='same',
                    activation=activation)
                if dropout:
                    x = tf.layers.dropout(x, keep_prob)
                if apply_batch_norm:
                    x = tf.contrib.layers.batch_norm(
                        x, is_training=is_training, decay=self.momentum)
            return x

    def fit(self, X, epochs=6000, batch_size=64):
        for i in range(epochs):
            train_d = True
            train_g = True
            keep_prob_train = 0.6
            for j in range(batch_size):
                # Creating noise
                n = np.random.uniform(0.0, 1.0,
                                      [batch_size, self.n_noise]).astype(np.float32)
                # Grabbing next batch
                batch_X = X[j * batch_size:(j + 1) * batch_size]
                batch_X = np.reshape(batch_X, newshape=[-1, 28, 28, 1])
                d_real_ls, d_fake_ls, g_ls, d_ls = self.sess.run(
                    [self.loss_d_real, self.loss_d_fake,
                     self.loss_g, self.loss_d],
                    feed_dict={self.real_images: batch_X,
                               self.noise: n,
                               self.keep_prob: keep_prob_train,
                               self.is_training: True}
                )
                d_real_ls = np.mean(d_real_ls)
                d_fake_ls = np.mean(d_fake_ls)

                g_ls = g_ls
                d_ls = d_ls

                if g_ls * 1.5 < d_ls:
                    train_g = False
                    pass

                if d_ls * 2 < g_ls:
                    train_d = False
                    pass
                if train_d:
                    self.sess.run(
                        self.optimizer_d,
                        feed_dict={self.noise: n,
                                   self.real_images: batch_X,
                                   self.keep_prob: keep_prob_train,
                                   self.is_training: True})
                if train_g:
                    self.sess.run(
                        self.optimizer_g,
                        feed_dict={self.noise: n,
                                   self.keep_prob: keep_prob_train,
                                   self.is_training: True})


def mnist():
    _mnist = input_data.read_data_sets('./inputs/mnist')
    # Resetting default graph, starting from scratch
    tf.reset_default_graph()

    epochs = 6000
    batch_size = 64
    n_noise = 200
    learning_rate = 0.00015
    # mnist specific variables
    img_dim = 28
    num_colors = 1

    # The keep_prob variable will be used by our dropout layers,
    # which we introduce for more stable learning outcome
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    d_sizes = {
        'conv_layers': [
            [5, 64, 2, True, False],
            [5, 64, 1, True, False],
            [5, 64, 1, True, False]
        ],
        'dense_layers': [128]
    }

    d1 = 4
    d2 = 1
    g_sizes = {
        'd1': d1,  # dim
        'd2': d2,  # channels,
        'n_noise': n_noise,
        'is_training': is_training,
        'dense_layers': [
            [d1 * d1 * d2, True, True]
        ],
        'conv_layers': [
            [5, 64, 2, True, True],
            [5, 64, 2, True, True],
            [5, 64, 1, True, True],
            [5, 1, 1, False, False]
        ]
    }

    dcgan = DCGAN(
        img_dim,
        num_colors,
        d_sizes,
        g_sizes,
        keep_prob,
        learning_rate,
        n_noise
    )
    X = _mnist.train.images
    dcgan.fit(X, epochs=epochs, batch_size=batch_size)

