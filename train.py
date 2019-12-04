from architectures.conv import *
from architectures.fully_connected_models import *
from tensorflow.keras.layers import ReLU
import matplotlib.pyplot as plt
import os
import sys


USE_PARTIAL_DATA = True
TRAIN_LIM = 2000
TEST_LIM = 200
BATCH_SIZE = 128
NUM_EPOCHS = 1
USE_BATCH_NORM = False
NUM_CLASSES = 10
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.1
VERBOSE = True
PARAMS = {'tower0_conv1': 192,
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
          'verbose': VERBOSE
          }

if __name__ == '__main__':
  args = sys.argv[1:]

  try:
    dataset = args[0]
    if dataset == 'cifar10':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'cifar100':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
      NUM_CLASSES = 100
    elif dataset == 'mnist':
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
      raise ValueError('Unknown dataset: %s' % dataset)
  except IndexError:
    print('No dataset provided; using CIFAR-10')
    dataset = 'cifar10'
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  try:
    activation_str = args[1]
    if activation_str == 'relu':
      activation = ReLU
    elif activation_str == 'silu':
      activation = SiLULayer
    elif activation_str == 'dsilu':
      activation = DSiLULayer
    else:
      raise ValueError('Unknown activation function: %s' % activation_str)
  except IndexError:
    print('No activation function provided; using ReLU')
    activation_str = 'relu'
    activation = ReLU

  try:
    batch_norm = args[2]
    if batch_norm == 'bn':
      USE_BATCH_NORM = True
    else:
      batch_norm = 'no-bn'
  except IndexError:
    batch_norm = 'no-bn'

  try:
    full_model = args[3]

    if full_model == 'full':
      USE_PARTIAL_DATA = False
      model = FitNet4(num_classes=NUM_CLASSES,
                      activation=activation,
                      batch_norm=USE_BATCH_NORM,
                      verbose=VERBOSE,
                      dropout_rate=DROPOUT_RATE,
                      )
    else:
      full_model = 'partial'
      model = InceptionLayerV2(num_classes=NUM_CLASSES,
                               batch_norm=USE_BATCH_NORM,
                               activation=activation,
                               **PARAMS)
  except IndexError:
    full_model = 'partial'
    model = InceptionLayerV2(num_classes=NUM_CLASSES,
                             batch_norm=USE_BATCH_NORM,
                             activation=activation,
                             **PARAMS)

  try:
    epochs = int(args[4])
    NUM_EPOCHS = epochs
  except IndexError:
    pass

  if VERBOSE:
    print('-------------------------------------------')
    print('Dataset: ', dataset, '; batch size: ', BATCH_SIZE)
    print('Num classes: ', NUM_CLASSES)
    print('learning rate: ', LEARNING_RATE, '; dropout rate: ', DROPOUT_RATE)
    print('Using activation function: ', activation_str)
    print('Use batch norm: ', str(USE_BATCH_NORM))
    print('Training %s model for %d epochs' % (full_model, NUM_EPOCHS))
    print('-------------------------------------------')

  x_train, x_test = x_train / 255.0, x_test / 255.0

  if USE_PARTIAL_DATA:
    print('Training on %d examples; validating on %d examples' % (TRAIN_LIM, TEST_LIM))
    x_train, y_train = x_train[:TRAIN_LIM], y_train[:TRAIN_LIM]
    x_test, y_test = x_test[:TEST_LIM], y_test[:TEST_LIM]
  else:
    print('Training on %d examples; validating on %d examples' % (x_train.shape[0],
                                                                  x_test.shape[0]))

  # tensorboard callback
  dir_path = os.path.dirname(os.path.abspath(__file__))
  run_time = time.strftime('%d%m%Y-%H:%M:%S')
  log_dir = dir_path + '/.logs/' + dataset + '-' + activation_str + '-' + run_time + '-' + batch_norm
  print('Tensorboard log directory: ', log_dir)

  lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        histogram_freq=1,
                                                        update_freq='batch')
  loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  metric = tf.keras.metrics.SparseCategoricalAccuracy()

  model.compile(optimizer=optimizer, loss=loss_function, metrics=[metric])
  start_time = time.time()

  history = model.fit(x_train, y_train,
                      batch_size=BATCH_SIZE,
                      epochs=NUM_EPOCHS,
                      validation_data=(x_test, y_test),
                      callbacks=[tensorboard_callback, lr_callback]
                      )

  elapsed_time = time.time() - start_time
  elapsed_hrs = int(elapsed_time // 3600)
  elapsed_min = int((elapsed_time - elapsed_hrs * 3600) // 60)
  elapsed_sec = int(elapsed_time - elapsed_hrs * 3600 - elapsed_min * 60)

  print(model.summary())
  print('---------------------------------------')
  print('elapsed time: %dh, %dm, %ds' % (elapsed_hrs, elapsed_min, elapsed_sec))
  print(history.history)

  fig, ax = plt.subplots(2, 1, figsize=(6, 7))
  ax[0].plot(history.history['val_sparse_categorical_accuracy'])
  ax[0].set_ylabel('Validation Accuracy')
  ax[1].plot(history.history['val_loss'])
  ax[1].set_ylabel('Validation Loss')
  plt.show()

  filename = dir_path + '/.results/' + activation_str + '.pickle'
  if os.path.exists(filename):
    with open(filename, 'rb') as f:
      all_data = pickle.load(f)
    all_data[run_time] = (history.history, run_time)
  else:
    all_data = {run_time: (history.history, run_time)}
  with open(filename, 'wb') as f:
    pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)
