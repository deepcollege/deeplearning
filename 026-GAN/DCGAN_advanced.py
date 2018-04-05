import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('./inputs/mnist')
# Resetting default graph, starting from scratch
tf.reset_default_graph()

epochs = 6000
batch_size = 64
n_noise = 200
learning_rate = 0.00015

real_images = tf.placeholder(
    dtype=tf.float32, shape=[None, 28, 28, 1], name='real_images')

# The keep_prob variable will be used by our dropout layers, which we introduce for more stable learning outcome
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


class DCGAN:
    def __init__(self,
                 img_length,
                 num_colors,
                 d_sizes,
                 g_sizes,
                 keep_prob):

        self.img_length = img_length
        self.num_colors = num_colors
        self.latent_dims = g_sizes['z']
        self.momentum = 0.99

        # For mnist, img_length = 28, num_colors = 1
        self.real_images = tf.placeholder(
            dtype=tf.float32,
            shape=[None, img_length, img_length, num_colors],
            name='real_images')

        noise = tf.placeholder(
            dtype=tf.float32, shape=[None, self.latent_dims])

        g = self.build_generator(
            noise, keep_prob, g_sizes=g_sizes)
        d_real = self.build_discriminator(
            self.real_images, reuse=None, keep_prob=keep_prob, d_sizes=d_sizes)
        d_fake = self.build_discriminator(
            g, reuse=True, keep_prob=keep_prob, d_sizes=d_sizes)


    def lrelu(self, x):
        return tf.maximum(x, tf.multiply(x, 0.2))

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
                    x = tf.layers.dropout(x, self.keep_prob)

                if apply_batch_norm:
                    x = tf.contrib.layers.batch_norm(
                        x, is_training=is_training, decay=self.momentum)
            for units in d_sizes['dense_layers']:
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
                    x = tf.layers.dropout(x, self.keep_prob)
                if apply_batch_norm:
                    x = tf.contrib.layers.batch_norm(
                        x, is_training=is_training, decay=self.momentum)
            return x
