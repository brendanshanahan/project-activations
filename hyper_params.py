import sys
from os import path
from architectures.fully_connected_models import *
import pickle


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

  if path.exists('results/cifar10-all-trials.pickle'):
    with open('results/cifar10-all-trials.pickle', 'rb') as f:
      all_runs = pickle.load(f)
    all_runs[run_name] = history.history
  else:
    all_runs = {run_name: history.history}
  with open('results/cifar10-all-trials.pickle', 'wb') as f:
    pickle.dump(all_runs, f, pickle.HIGHEST_PROTOCOL)
