import tensorflow as tf
import sys
from os import path
from tensorboard.plugins.hparams import api as hp
from architectures.fully_connected_models import *
import pickle
import math


class NBatchLogger(tf.keras.callbacks.TensorBoard):
  def __init__(self, log_every=1, **kwargs):
    super(NBatchLogger, self).__init__(**kwargs)
    self.log_every = log_every
    self.counter = 0

  def on_batch_end(self, batch, logs=None):
    self.counter += 1
    if self.counter % self.log_every == 0:
      for name, value in logs.items():
        if name in ['batch', 'size']:
          continue
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value.item()
        summary_value.tag = name
        self.writer.add_summary(summary, self.counter)
      self.writer.flush()

    super().on_batch_end(batch, logs)

  def on_epoch_end(self, epoch, logs=None):
    for name, value in logs.items():
      if (name in ['batch', 'size']) or ('val' not in name):
        continue
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value.item()
      summary_value.tag = name
      self.writer.add_summary(summary, epoch)
    self.writer.flush()


if __name__ == "__main__":
  layer_size = int(sys.argv[1])
  learning_rate = float(sys.argv[2])
  batch_size = int(sys.argv[3])
  num_epochs = int(sys.argv[4])

  try:
    act = sys.argv[5]
    data = sys.argv[6]
  except IndexError:
    act = 'relu'
    data = 'mnist'

  try:
    batch_norm = sys.argv[7]
    if batch_norm == 'bn':
      batch_norm = True
    else:
      batch_norm = False
  except IndexError:
    batch_norm = False

  if act == 'silu':
    activation = silu
  elif act == 'dsilu':
    activation = dsilu
  elif act == 'relu':
    activation = tf.nn.relu
  else:
    print('Unknown activation function %s - using ReLU' % act)
    activation = tf.nn.relu

  if data == 'cifar10':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

  if batch_norm:
    run_name = '3fc-bn-' + act + "-" + data + "-%d-hu-%f-lr-%d-bs" % (layer_size, learning_rate, batch_size)
    model = ThreeLayerFCBatchNorm([layer_size] * 3, activation=activation)
  else:
    run_name = '3fc-' + act + "-" + data + "-%d-hu-%f-lr-%d-bs" % (layer_size, learning_rate, batch_size)
    model = ThreeLayerFC([layer_size] * 3, activation=activation)

  log_dir = 'full-batches/' + run_name
  print('logging to directory ', log_dir)

  loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_2=0.995, epsilon=1e-4)
  metric = tf.keras.metrics.SparseCategoricalAccuracy()
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                            histogram_freq=1,
                                            )

  model.compile(optimizer=optimizer,
                loss=loss_function,
                metrics=[metric]
                )

  history = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      validation_data=(x_test, y_test),
                      callbacks=[callback]
                      )

  # epoch_batch = math.ceil(x_train.shape[0] / 4)
  # steps_per_epoch = math.ceil(epoch_batch / batch_size)
  #
  # print('steps per epoch: ', steps_per_epoch)
  # print('epoch batch size: ', epoch_batch)
  #
  # history = model.fit(x_train, y_train,
  #                     batch_size=batch_size,
  #                     epochs=num_epochs*4,
  #                     steps_per_epoch=steps_per_epoch,
  #                     validation_data=(x_test, y_test),
  #                     callbacks=[callback]
  #                     )
  #
  # print('training history: ', history.history)
  #
  # if path.exists('final-runs.pickle'):
  #   with open('final-runs.pickle', 'rb') as f:
  #     all_runs = pickle.load(f)
  #   all_runs[run_name] = history.history
  # else:
  #   all_runs = {run_name: history.history}
  # with open('final-runs.pickle', 'wb') as f:
  #   pickle.dump(all_runs, f, pickle.HIGHEST_PROTOCOL)
