import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import ReLU
# from fully_connected_models import SiLULayer, DSiLULayer
import math
import time
import pickle


class InceptionModule(Model):
  def __init__(self, **kwargs):
    super(InceptionModule, self).__init__()

    self.batch_norm = kwargs.pop('batch_norm', False)
    activation = kwargs.pop('activation', ReLU)
    tower0_conv1_filters = kwargs.pop('tower0_conv1', 64)
    tower1_conv1_filters = kwargs.pop('tower1_conv1', 64)
    tower1_conv2_filters = kwargs.pop('tower1_conv2', 64)
    tower1_conv3_filters = kwargs.pop('tower1_conv3', 64)
    tower2_conv1_filters = kwargs.pop('tower2_conv1', 64)
    tower2_conv2_filters = kwargs.pop('tower2_conv2', 64)
    tower3_conv1_filters = kwargs.pop('tower3_conv1', 64)
    verbose = kwargs.get('verbose', False)
    name = kwargs.get('name', 'NA')

    if verbose:
      print('--------------- Inception Layer: %s ---------------' % name)
      print('tower 0, conv 1 filters: ', tower0_conv1_filters)
      print('tower 1, conv 1 filters: ', tower1_conv1_filters)
      print('tower 1, conv 2 filters: ', tower1_conv2_filters)
      print('tower 1, conv 3 filters: ', tower1_conv3_filters)
      print('tower 2, conv 1 filters: ', tower2_conv1_filters)
      print('tower 2, conv 2 filters: ', tower2_conv2_filters)
      print('tower 3, conv 1 filters: ', tower3_conv1_filters)

    self.tower0_conv1 = Conv2D(filters=tower0_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower0_act1 = activation()

    self.tower1_conv1 = Conv2D(filters=tower1_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower1_act1 = activation()
    self.tower1_conv2 = Conv2D(filters=tower1_conv2_filters, kernel_size=(3, 3), padding='same')
    self.tower1_act2 = activation()
    self.tower1_conv3 = Conv2D(filters=tower1_conv3_filters, kernel_size=(3, 3), padding='same')
    self.tower1_act3 = activation()

    self.tower2_conv1 = Conv2D(filters=tower2_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower2_act1 = activation()
    self.tower2_conv2 = Conv2D(filters=tower2_conv2_filters, kernel_size=(3, 3), padding='same')
    self.tower2_act2 = activation()

    self.tower3_pool1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
    # self.tower3_pool1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    self.tower3_conv1 = Conv2D(filters=tower3_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower3_act1 = activation()

    # self.flat = Flatten()
    # self.fc1 = Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

    if self.batch_norm:
      self.tower0_bn1 = BatchNormalization()
      self.tower1_bn1 = BatchNormalization()
      self.tower1_bn2 = BatchNormalization()
      self.tower2_bn1 = BatchNormalization()
      self.tower2_bn2 = BatchNormalization()
      self.tower3_bn1 = BatchNormalization()

  def call(self, x, **kwargs):
    training = kwargs.get('training', False)

    # print('x shape: ', x.shape)

    # tower 0 feed-forward
    x0 = self.tower0_conv1(x)
    if self.batch_norm:
      x0 = self.tower0_bn1(x0)
    x0 = self.tower0_act1(x0)

    # tower one feed-forward
    x1 = self.tower1_conv1(x)
    if self.batch_norm:
      x1 = self.tower1_bn1(x1)
    x1 = self.tower1_act1(x1)
    x1 = self.tower1_conv2(x1)
    if self.batch_norm:
      x1 = self.tower1_bn2(x1)
    x1 = self.tower1_act2(x1)
    x1 = self.tower1_conv3(x1)
    if self.batch_norm:
      x1 = self.tower1_bn3(x1)
    x1 = self.tower1_act3(x1)

    # tower two feed-forward
    x2 = self.tower2_conv1(x)
    if self.batch_norm:
      x2 = self.tower2_bn1(x2)
    x2 = self.tower2_act1(x2)
    x2 = self.tower2_conv2(x2)
    if self.batch_norm:
      x2 = self.tower2_bn2(x2)
    x2 = self.tower2_act2(x2)

    # tower three feed-forward
    x3 = self.tower3_pool1(x)
    x3 = self.tower3_conv1(x3)
    if self.batch_norm:
      x3 = self.tower3_bn1(x3)
    x3 = self.tower3_act1(x3)

    print('x0 shape: ', x0.shape)
    print('x1 shape: ', x1.shape)
    print('x2 shape: ', x2.shape)
    print('x3 shape: ', x3.shape)

    out = concatenate([x0, x1, x2, x3], axis=3)
    # out = self.flat(out)
    # out = self.fc1(out)
    return out


class InceptionModuleV2(Model):
  def __init__(self, **kwargs):
    super(InceptionModuleV2, self).__init__()

    self.batch_norm = kwargs.pop('batch_norm', False)
    activation = kwargs.pop('activation', ReLU)
    tower0_conv1_filters = kwargs.pop('tower0_conv1', 64)
    tower1_conv1_filters = kwargs.pop('tower1_conv1', 64)
    tower1_conv2_filters = kwargs.pop('tower1_conv2', 64)
    tower1_conv3_filters = kwargs.pop('tower1_conv3', 64)
    tower1_conv4_filters = kwargs.pop('tower1_conv4', 64)
    tower1_conv5_filters = kwargs.pop('tower1_conv5', 64)
    tower2_conv1_filters = kwargs.pop('tower2_conv1', 64)
    tower2_conv2_filters = kwargs.pop('tower2_conv2', 64)
    tower2_conv3_filters = kwargs.pop('tower2_conv3', 64)
    tower3_conv1_filters = kwargs.pop('tower3_conv1', 64)
    verbose = kwargs.get('verbose', False)
    name = kwargs.get('name', 'NA')

    if verbose:
      print('--------------- Inception Layer: %s ---------------' % name)
      print('tower 0, conv 1 filters: ', tower0_conv1_filters)
      print('tower 1, conv 1 filters: ', tower1_conv1_filters)
      print('tower 1, conv 2 filters: ', tower1_conv2_filters)
      print('tower 1, conv 3 filters: ', tower1_conv3_filters)
      print('tower 1, conv 4 filters: ', tower1_conv4_filters)
      print('tower 1, conv 5 filters: ', tower1_conv5_filters)
      print('tower 2, conv 1 filters: ', tower2_conv1_filters)
      print('tower 2, conv 2 filters: ', tower2_conv2_filters)
      print('tower 2, conv 3 filters: ', tower2_conv3_filters)
      print('tower 3, conv 1 filters: ', tower3_conv1_filters)

    self.tower0_conv1 = Conv2D(filters=tower0_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower0_act1 = activation()

    self.tower1_conv1 = Conv2D(filters=tower1_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower1_act1 = activation()
    self.tower1_conv2 = Conv2D(filters=tower1_conv2_filters, kernel_size=(1, 5), padding='same')
    self.tower1_act2 = activation()
    self.tower1_conv3 = Conv2D(filters=tower1_conv3_filters, kernel_size=(5, 1), padding='same')
    self.tower1_act3 = activation()
    self.tower1_conv4 = Conv2D(filters=tower1_conv4_filters, kernel_size=(1, 5), padding='same')
    self.tower1_act4 = activation()
    self.tower1_conv5 = Conv2D(filters=tower1_conv5_filters, kernel_size=(5, 1), padding='same')
    self.tower1_act5 = activation()

    self.tower2_conv1 = Conv2D(filters=tower2_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower2_act1 = activation()
    self.tower2_conv2 = Conv2D(filters=tower2_conv2_filters, kernel_size=(1, 3), padding='same')
    self.tower2_act2 = activation()
    self.tower2_conv3 = Conv2D(filters=tower2_conv3_filters, kernel_size=(3, 1), padding='same')
    self.tower2_act3 = activation()

    self.tower3_pool1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')
    # self.tower3_pool1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
    self.tower3_conv1 = Conv2D(filters=tower3_conv1_filters, kernel_size=(1, 1), padding='same')
    self.tower3_act1 = activation()

    # self.flat = Flatten()
    # self.fc1 = Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

    if self.batch_norm:
      self.tower0_bn1 = BatchNormalization()
      self.tower1_bn1 = BatchNormalization()
      self.tower1_bn2 = BatchNormalization()
      self.tower1_bn3 = BatchNormalization()
      self.tower1_bn4 = BatchNormalization()
      self.tower1_bn5 = BatchNormalization()
      self.tower2_bn1 = BatchNormalization()
      self.tower2_bn2 = BatchNormalization()
      self.tower2_bn3 = BatchNormalization()
      self.tower3_bn1 = BatchNormalization()

  def call(self, x, **kwargs):
    training = kwargs.get('training', False)

    # print('x shape: ', x.shape)

    # tower 0 feed-forward
    x0 = self.tower0_conv1(x)
    if self.batch_norm:
      x0 = self.tower0_bn1(x0)
    x0 = self.tower0_act1(x0)

    # tower one feed-forward
    x1 = self.tower1_conv1(x)
    if self.batch_norm:
      x1 = self.tower1_bn1(x1)
    x1 = self.tower1_act1(x1)
    # 2
    x1 = self.tower1_conv2(x1)
    if self.batch_norm:
      x1 = self.tower1_bn2(x1)
    x1 = self.tower1_act2(x1)
    # 3
    x1 = self.tower1_conv3(x1)
    if self.batch_norm:
      x1 = self.tower1_bn3(x1)
    x1 = self.tower1_act3(x1)
    # 4
    x1 = self.tower1_conv4(x1)
    if self.batch_norm:
      x1 = self.tower1_bn4(x1)
    x1 = self.tower1_act4(x1)
    # 5
    x1 = self.tower1_conv5(x1)
    if self.batch_norm:
      x1 = self.tower1_bn5(x1)
    x1 = self.tower1_act5(x1)

    # tower two feed-forward
    x2 = self.tower2_conv1(x)
    if self.batch_norm:
      x2 = self.tower2_bn1(x2)
    x2 = self.tower2_act1(x2)
    x2 = self.tower2_conv2(x2)
    if self.batch_norm:
      x2 = self.tower2_bn2(x2)
    x2 = self.tower2_act2(x2)
    x2 = self.tower2_conv3(x2)
    if self.batch_norm:
      x2 = self.tower2_bn3(x2)
    x2 = self.tower2_act3(x2)

    # tower three feed-forward
    x3 = self.tower3_pool1(x)
    x3 = self.tower3_conv1(x3)
    if self.batch_norm:
      x3 = self.tower3_bn1(x3)
    x3 = self.tower3_act1(x3)

    out = concatenate([x0, x1, x2, x3], axis=3)
    # out = self.flat(out)
    # out = self.fc1(out)
    return out


class InceptionLayerV2(Model):
  def __init__(self, **kwargs):
    super(InceptionLayerV2, self).__init__()
    self.batch_norm = kwargs.get('batch_norm', False)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    num_classes = kwargs.get('num_classes', 100)
    activation = kwargs.get('activation', ReLU)

    self.conv1 = Conv2D(filters=192, kernel_size=(1, 1), padding='same')
    self.act1 = activation()

    incep1_params = {'tower0_conv1': 320,
                     'tower1_conv1': 384,
                     'tower1_conv2': 384,
                     'tower1_conv3': 384,
                     'tower1_conv4': 384,
                     'tower1_conv5': 384,
                     'tower2_conv1': 448,
                     'tower2_conv2': 384,
                     'tower2_conv3': 384,
                     'tower3_conv1': 448,
                     'name': 'incep_v2',
                     }

    self.incep1 = InceptionModuleV2(**incep1_params)
    # self.incep1 = InceptionModuleV2(**kwargs)
    # self.incep1 = InceptionModuleV2()

    self.conv2 = Conv2D(filters=192, kernel_size=(1, 1), padding='same')
    self.act2 = activation()

    self.flat = Flatten()
    self.fc = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    if self.batch_norm:
      self.bn1 = BatchNormalization()
      self.bn2 = BatchNormalization()

  def call(self, x, **kwargs):
    training = kwargs.pop('training', False)

    # feed-forward layers
    x = self.conv1(x)
    if self.batch_norm:
      x = self.bn1(x)
    x = self.act1(x)
    x = self.incep1(x)
    x = self.conv2(x)
    if self.batch_norm:
      x = self.bn2(x)
    x = self.act2(x)
    x = self.flat(x)
    scores = self.fc(x)
    return scores


class ConvolutionalModel(Model):
  def __init__(self, **kwargs):
    super(ConvolutionalModel, self).__init__()
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    self.batch_norm = kwargs.get('batch_norm', False)
    num_classes = kwargs.get('num_classes', 100)
    activation = kwargs.get('activation', ReLU)
    verbose = kwargs.get('verbose', False)

    # convolution filter sizes
    conv1_filters = kwargs.get('conv1_filters', 64)
    conv2_filters = kwargs.get('conv2_filters', 92)
    conv3_filters = kwargs.get('conv1_filters', 128)
    conv4_filters = kwargs.get('conv2_filters', 192)
    conv5_filters = kwargs.get('conv1_filters', 192)

    # initialize layers
    self.conv1 = Conv2D(filters=conv1_filters, kernel_size=(3, 3), padding='same')
    self.act1 = activation()

    self.conv2 = Conv2D(filters=conv2_filters, kernel_size=(3, 3), padding='same')
    self.act2 = activation()

    self.conv3 = Conv2D(filters=conv3_filters, kernel_size=(3, 3), padding='same')
    self.act3 = activation()

    self.pool1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')

    self.conv4 = Conv2D(filters=conv4_filters, kernel_size=(1, 1), padding='same')
    self.act4 = activation()

    self.pool2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')

    incep1_params = {'tower0_conv1': 64,
                     'tower1_conv1': 64,
                     'tower1_conv2': 64,
                     'tower1_conv3': 96,
                     'tower2_conv1': 64,
                     'tower2_conv2': 64,
                     'tower3_conv1': 96,
                     'name': 'incep1_v1',
                     'verbose': verbose
                     }

    incep2_params = {'tower0_conv1': 128,
                     'tower1_conv1': 128,
                     'tower1_conv2': 128,
                     'tower1_conv3': 192,
                     'tower2_conv1': 128,
                     'tower2_conv2': 192,
                     'tower3_conv1': 192,
                     'name': 'incep2_v1',
                     'verbose': verbose
                     }

    incep3_params = {'tower0_conv1': 192,
                     'tower1_conv1': 128,
                     'tower1_conv2': 128,
                     'tower1_conv3': 128,
                     'tower1_conv4': 192,
                     'tower1_conv5': 192,
                     'tower2_conv1': 128,
                     'tower2_conv2': 192,
                     'tower2_conv3': 192,
                     'tower3_conv1': 192,
                     'name': 'incep3_v2',
                     'verbose': verbose
                     }

    incep4_params = {'tower0_conv1': 320,
                     'tower1_conv1': 384,
                     'tower1_conv2': 384,
                     'tower1_conv3': 384,
                     'tower1_conv4': 384,
                     'tower1_conv5': 384,
                     'tower2_conv1': 448,
                     'tower2_conv2': 384,
                     'tower2_conv3': 384,
                     'tower3_conv1': 448,
                     'name': 'incep4_v2',
                     'verbose': verbose
                     }

    self.incep1 = InceptionModule(**incep1_params)
    self.incep2 = InceptionModule(**incep2_params)

    self.pool3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')

    self.incep3 = InceptionModuleV2(**incep3_params)
    self.incep4 = InceptionModuleV2(**incep4_params)

    self.pool4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')

    self.conv5 = Conv2D(filters=conv5_filters, kernel_size=(1, 1), padding='same')
    self.act5 = activation()

    self.flat = Flatten()

    # self.fc1 = Dense(fc1_size, kernel_initializer=initializer)
    # self.act6 = activation()
    #
    # self.fc2 = Dense(fc2_size, kernel_initializer=initializer)
    # self.act7 = activation()

    self.fc3 = Dense(num_classes, activation='softmax', kernel_initializer=initializer)

    if self.batch_norm:
      self.bn1 = BatchNormalization()
      self.bn2 = BatchNormalization()
      self.bn3 = BatchNormalization()
      self.bn4 = BatchNormalization()
      self.bn5 = BatchNormalization()
      # self.bn6 = BatchNormalization()

  def call(self, x, **kwargs):
    training = kwargs.pop('training', False)

    # feed-forward layers
    x = self.conv1(x)
    if self.batch_norm:
      x = self.bn1(x)
    x = self.act1(x)

    x = self.conv2(x)
    if self.batch_norm:
      x = self.bn2(x)
    x = self.act2(x)

    x = self.pool1(x)

    x = self.conv3(x)
    if self.batch_norm:
      x = self.bn3(x)
    x = self.act3(x)

    x = self.conv4(x)
    if self.batch_norm:
      x = self.bn4(x)
    x = self.act4(x)

    x = self.pool2(x)

    # inception modules (1st block)
    x = self.incep1(x)
    x = self.incep2(x)

    # pool
    x = self.pool3(x)

    # inception modules (2nd block)
    x = self.incep3(x)
    x = self.incep4(x)

    x = self.pool4(x)

    x = self.conv5(x)
    if self.batch_norm:
      x = self.bn5(x)
    x = self.act5(x)

    x = self.flat(x)

    # x = self.fc1(x)
    # if self.batch_norm:
    #   x = self.bn6(x)
    # x = self.act6(x)
    #
    # x = self.fc2(x)
    # if self.batch_norm:
    #   x = self.bn7(x)
    # x = self.act7(x)

    scores = self.fc3(x)

    return scores


if __name__ == '__main__':

  dataset = 'cifar10'
  use_partial_data = True

  train_lim = 1000
  test_lim = 100

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  # if dataset == 'cifar100':
  #   print('########## training on CIFAR-100 ##########')
  #   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
  #   x_train, x_test = x_train / 255.0, x_test / 255.0
  #
  #   model = ConvolutionalModel(num_classes=100, verbose=True)
  #
  # else:
  #   print('########## training on CIFAR-10 ##########')
  #   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  #   x_train, x_test = x_train / 255.0, x_test / 255.0
  #
  #   model = ConvolutionalModel(num_classes=10, verbose=True)

  if use_partial_data:
    x_train, y_train = x_train[:train_lim], y_train[:train_lim]
    x_test, y_test = x_test[:test_lim], y_test[:test_lim]

  # use half of the test set as the validaiton set
  # x_val, y_val = x_test[:test_lim], y_test[:test_lim]
  # x_test, y_test = x_test[test_lim:], y_test[test_lim:]

  print('----------------- running model -----------------')
  print('x_train shape: ', x_train.shape)
  print('x_test shape: ', x_test.shape)
  print('----------------------------------------')
  print('y_train shape: ', y_train.shape)
  print('y_test shape: ', y_test.shape)
  print('----------------------------------------')

  # params = {'tower0_conv1': 192,
  #           'tower1_conv1': 128,
  #           'tower1_conv2': 128,
  #           'tower1_conv3': 192,
  #           'tower2_conv1': 128,
  #           'tower2_conv2': 192,
  #           'tower3_conv1': 192,
  #           'name': 'incep3',
  #           'verbose': False
  #           }

  params = {'tower0_conv1': 192,
            'tower1_conv1': 128,
            'tower1_conv2': 128,
            'tower1_conv3': 128,
            'tower1_conv4': 192,
            'tower1_conv5': 192,
            'tower2_conv1': 128,
            'tower2_conv2': 192,
            'tower2_conv3': 192,
            'tower3_conv1': 192,
            'name': 'incepv2',
            'verbose': False
            }

  # model = InceptionModuleV2(**params)
  model = ConvolutionalModel(num_classes=10, verbose=True)

  loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()
  metric = tf.keras.metrics.SparseCategoricalAccuracy()

  # tensorboard callback
  run_time = time.strftime('%d%m%Y-%H:%M:%S')
  log_dir = '/home/brendan/nn/project/logs/' + dataset + '-' + run_time
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  model.compile(optimizer=optimizer, loss=loss_function, metrics=[metric])

  batch_size = 128
  num_epochs = 1
  start_time = time.time()

  history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      validation_data=(x_test, y_test),
                      callbacks=[callback]
                      )

  elapsed_time = time.time() - start_time
  elapsed_hrs = int(elapsed_time // 3600)
  elapsed_min = int((elapsed_time - elapsed_hrs * 3600) // 60)
  elapsed_sec = int(elapsed_time - elapsed_hrs * 3600 - elapsed_min * 60)

  print(model.summary())
  print('---------------------')
  print('elapsed time: %dh, %dm, %ds' % (elapsed_hrs, elapsed_min, elapsed_sec))
  print('--------------- history ---------------')
  print(history.history)

  filename = dataset + '-' + run_time + '-' + 'mini.pickle'
  with open(filename, 'wb') as f:
    pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

