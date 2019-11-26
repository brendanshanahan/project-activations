import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ReLU
import matplotlib.pyplot as plt
# from tensorflow._api.v2.keras import sigmoid

def silu(x):
    """
    sigmoid-weighted linear unit activation function as described in (...)

    :param x: input
    :return: activations
    """
    return x * sigmoid(x)


def dsilu(x):
    """
    derivative of sigmoid-weighted linear unit activation function as
    described in Elfwing et. al., Neural Networks 107 (2018) 3–11

    :param x: input
    :return: activations
    """
    return sigmoid(x) * (1 + x * (1 - sigmoid(x)))


class SiLULayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SiLULayer, self).__init__()

    def call(self, x):
        return silu(x)


class DSiLULayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DSiLULayer, self).__init__()

    def call(self, x):
        return dsilu(x)


class FullyConnectedNet(tf.keras.Model):
    pass


class ThreeLayerFC(FullyConnectedNet):
    def __init__(self, layer_sizes, activation='relu', **params):
        super().__init__()

        l1_size, l2_size, num_classes = layer_sizes
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.flatten = Flatten()
        self.fc1 = Dense(l1_size, activation=activation, kernel_initializer=initializer)
        self.fc2 = Dense(l2_size, activation=activation, kernel_initializer=initializer)
        self.fc3 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    def call(self, x, training=None):
        x = self.flatten(x)
        l1_out = self.fc1(x)
        l2_out = self.fc2(l1_out)
        scores = self.fc3(l2_out)

        return scores


class ThreeLayerFCBatchNorm(FullyConnectedNet):
    def __init__(self, layer_sizes, activation='relu', **params):
        super().__init__()

        l1_size, l2_size, num_classes = layer_sizes
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        if activation == 'silu':
            self.act1 = SiLULayer()
            self.act2 = SiLULayer()
            # activation = SiLULayer
        elif activation == 'dsilu':
            self.act1 = DSiLULayer()
            self.act2 = DSiLULayer()
            # activation = DSiLULayer
        elif activation == 'relu':
            self.act1 = ReLU()
            self.act2 = ReLU()
            # activation = ReLU
        else:
            print('Unrecognized activaton function; using ReLU')
            self.act1 = ReLU()
            self.act2 = ReLU()
            # activation = tf.nn.relu

        self.flatten = Flatten()
        self.fc1 = Dense(l1_size, kernel_initializer=initializer)
        self.bn1 = BatchNormalization()
        # self.act1 = activation
        self.fc2 = Dense(l2_size, kernel_initializer=initializer)
        self.bn2 = BatchNormalization()
        # self.act2 = activation
        self.fc3 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    def call(self, x, training=None):
        x = self.flatten(x)
        l1_out = self.fc1(x)
        l2_out = self.bn1(l1_out)
        l3_out = self.act1(l2_out)
        l4_out = self.fc2(l3_out)
        l5_out = self.bn2(l4_out)
        l6_out = self.act2(l5_out)
        scores = self.fc3(l6_out)

        return scores


class FiveLayerFC(FullyConnectedNet):
    def __init__(self, layer_sizes, activation='relu', **params):
        super().__init__()

        l1_size, l2_size, l3_size, l4_size, num_classes = layer_sizes
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.flatten = Flatten()
        self.fc1 = Dense(l1_size, activation=activation, kernel_initializer=initializer)
        self.fc2 = Dense(l2_size, activation=activation, kernel_initializer=initializer)
        self.fc3 = Dense(l3_size, activation=activation, kernel_initializer=initializer)
        self.fc4 = Dense(l4_size, activation=activation, kernel_initializer=initializer)
        self.fc5 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    def call(self, x, training=None):
        x = self.flatten(x)
        l1_out = self.fc1(x)
        l2_out = self.fc2(l1_out)
        l3_out = self.fc3(l2_out)
        l4_out = self.fc4(l3_out)
        scores = self.fc5(l4_out)

        return scores


class FiveLayerFCBatchNorm(FullyConnectedNet):
    def __init__(self, layer_sizes, activation='relu', **params):
        super().__init__()

        l1_size, l2_size, l3_size, l4_size, num_classes = layer_sizes
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

        self.flatten = Flatten()
        self.fc1 = Dense(l1_size, activation=activation, kernel_initializer=initializer)
        self.fc2 = Dense(l2_size, activation=activation, kernel_initializer=initializer)
        self.fc3 = Dense(l3_size, activation=activation, kernel_initializer=initializer)
        self.fc4 = Dense(l4_size, activation=activation, kernel_initializer=initializer)
        self.fc5 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    def call(self, x, training=None):
        x = self.flatten(x)
        l1_out = self.fc1(x)
        l2_out = self.fc2(l1_out)
        l3_out = self.fc3(l2_out)
        l4_out = self.fc4(l3_out)
        scores = self.fc5(l4_out)

        return scores
