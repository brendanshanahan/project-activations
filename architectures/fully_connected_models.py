import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, ReLU


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
  described in Elfwing et. al., Neural Networks 107 (2018) 3-11

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


class FunctionApprox(Model):
  def __init__(self, **kwargs):
    super(FunctionApprox, self).__init__()

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    activation = kwargs.get('activation', ReLU)
    self.batch_norm = kwargs.get('batch_norm', False)

    # layer sizes
    layer1_units = kwargs.get('layer1_units', 20)
    layer2_units = kwargs.get('layer2_units', 20)
    layer3_units = kwargs.get('layer3_units', 20)
    layer4_units = kwargs.get('layer4_units', 20)

    self.fc1 = Dense(units=layer1_units, kernel_initializer=initializer)
    self.act1 = activation()
    self.fc2 = Dense(units=layer2_units, kernel_initializer=initializer)
    self.act2 = activation()
    self.fc3 = Dense(units=layer3_units, kernel_initializer=initializer)
    self.act3 = activation()
    self.fc4 = Dense(units=layer4_units, kernel_initializer=initializer)
    self.act4 = activation()
    self.out = Dense(1)

    if self.batch_norm:
      self.bn1 = BatchNormalization()
      self.bn2 = BatchNormalization()
      self.bn3 = BatchNormalization()
      self.bn4 = BatchNormalization()

  def call(self, x, training=False):

    x = self.fc1(x)
    if self.batch_norm:
      x = self.bn1(x, training=training)
    x = self.act1(x)
    x = self.fc2(x)
    if self.batch_norm:
      x = self.bn2(x, training=training)
    x = self.act2(x)
    x = self.fc3(x)
    if self.batch_norm:
      x = self.bn3(x, training=training)
    x = self.act3(x)
    x = self.fc4(x)
    if self.batch_norm:
      x = self.bn4(x, training=training)
    x = self.act4(x)
    x = self.out(x)

    return x


class ThreeLayerFC(Model):
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


class ThreeLayerFCBatchNorm(Model):
  def __init__(self, layer_sizes, activation='relu', **params):
    super().__init__()

    l1_size, l2_size, num_classes = layer_sizes
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    if activation == 'silu':
      self.act1 = SiLULayer()
      self.act2 = SiLULayer()
    elif activation == 'dsilu':
      self.act1 = DSiLULayer()
      self.act2 = DSiLULayer()
    elif activation == 'relu':
      self.act1 = ReLU()
      self.act2 = ReLU()
    else:
      print('Unrecognized activaton function; using ReLU')
      self.act1 = ReLU()
      self.act2 = ReLU()

    self.flatten = Flatten()
    self.fc1 = Dense(l1_size, kernel_initializer=initializer)
    self.bn1 = BatchNormalization()
    self.fc2 = Dense(l2_size, kernel_initializer=initializer)
    self.bn2 = BatchNormalization()
    self.fc3 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

  def call(self, x, training=False):
    x = self.flatten(x)
    l1_out = self.fc1(x)
    l2_out = self.bn1(l1_out, training=training)
    l3_out = self.act1(l2_out)
    l4_out = self.fc2(l3_out)
    l5_out = self.bn2(l4_out, training=training)
    l6_out = self.act2(l5_out)
    scores = self.fc3(l6_out)

    return scores


class FiveLayerFC(Model):
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

  def call(self, x, training=False):
    x = self.flatten(x)
    l1_out = self.fc1(x)
    l2_out = self.fc2(l1_out)
    l3_out = self.fc3(l2_out)
    l4_out = self.fc4(l3_out)
    scores = self.fc5(l4_out)

    return scores


class FiveLayerFCBatchNorm(Model):
  def __init__(self, layer_sizes, activation='relu', **params):
    super().__init__()

    l1_size, l2_size, l3_size, l4_size, num_classes = layer_sizes
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    if activation == 'silu':
      self.act1 = SiLULayer()
      self.act2 = SiLULayer()
      self.act3 = SiLULayer()
      self.act4 = SiLULayer()
    elif activation == 'dsilu':
      self.act1 = DSiLULayer()
      self.act2 = DSiLULayer()
      self.act3 = DSiLULayer()
      self.act4 = DSiLULayer()
    elif activation == 'relu':
      self.act1 = ReLU()
      self.act2 = ReLU()
      self.act3 = ReLU()
      self.act4 = ReLU()
    else:
      print('Unrecognized activaton function; using ReLU')
      self.act1 = ReLU()
      self.act2 = ReLU()
      self.act3 = ReLU()
      self.act4 = ReLU()

    self.flatten = Flatten()
    self.fc1 = Dense(l1_size, kernel_initializer=initializer)
    self.bn1 = BatchNormalization()
    self.fc2 = Dense(l2_size, kernel_initializer=initializer)
    self.bn2 = BatchNormalization()
    self.fc3 = Dense(l3_size, kernel_initializer=initializer)
    self.bn3 = BatchNormalization()
    self.fc4 = Dense(l4_size, kernel_initializer=initializer)
    self.bn4 = BatchNormalization()
    self.fc5 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

  def call(self, x, training=False):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.bn1(x, training=training)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.bn2(x, training=training)
    x = self.act2(x)
    x = self.fc3(x)
    x = self.bn3(x, training=training)
    x = self.act3(x)
    x = self.fc4(x)
    x = self.bn4(x, training=training)
    x = self.act4(x)
    scores = self.fc5(x)

    return scores
