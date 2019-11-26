import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import pickle
from architectures.fully_connected_models import silu, dsilu
from tensorflow.keras.backend import relu, sigmoid


def plot_val_accuracy(run='results/final-runs.pickle'):
  with open(run, 'rb') as f:
    all_data = pickle.load(f)

  cifar10_data = [k for k in all_data.keys() if 'cifar10' in k]
  mnist_data = [k for k in all_data.keys() if 'mnist' in k]

  fig, ax = plt.subplots(2, 3, figsize=(7, 4), sharex='col', sharey='row')
  fig.suptitle('Validation accuracy', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.86, wspace=0.12, hspace=0.17)

  for c in cifar10_data:
    label = 'with BN' if 'bn' in c else 'without BN'
    alpha = 1.0 if 'bn' in c else 0.4
    y = all_data[c]['val_sparse_categorical_accuracy']
    x = np.linspace(0, 15, num=len(y))
    if 'relu' in c:
      label = 'ReLU ' + label
      ax[0, 0].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 0].set_title('ReLU', fontsize=10)
    elif 'dsilu' in c:
      label = 'dSiLU ' + label
      ax[0, 2].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 2].set_title('DSiLU', fontsize=10)
    elif 'silu' in c:
      label = 'SiLU ' + label
      ax[0, 1].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 1].set_title('SiLU', fontsize=10)

  for m in mnist_data:
    label = 'with BN' if 'bn' in m else 'without BN'
    alpha = 1.0 if 'bn' in m else 0.4
    y = all_data[m]['val_sparse_categorical_accuracy']
    x = np.linspace(0, 15, num=len(y))
    if 'relu' in m:
      label = 'ReLU ' + label
      ax[1, 0].plot(x, y, color='black', label=label, alpha=alpha)
    elif 'dsilu' in m:
      ax[1, 2].plot(x, y, color='black', label=label, alpha=alpha)
    elif 'silu' in m:
      ax[1, 1].plot(x, y, color='black', label=label, alpha=alpha)

  plt.legend(loc='right')

  ax[0, 0].set_ylabel('CIFAR-10', fontsize=10)
  ax[1, 0].set_ylabel('MNIST', fontsize=10)
  plt.show()


def plot_val_loss(run='final-runs.pickle'):
  with open(run, 'rb') as f:
    all_data = pickle.load(f)

  cifar10_data = [k for k in all_data.keys() if 'cifar10' in k]
  mnist_data = [k for k in all_data.keys() if 'mnist' in k]

  fig, ax = plt.subplots(2, 3, figsize=(7, 4), sharex='col', sharey='row')
  fig.suptitle('Training loss', fontsize=16)
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.86, wspace=0.12, hspace=0.17)

  for c in cifar10_data:
    label = 'with BN' if 'bn' in c else 'without BN'
    alpha = 1.0 if 'bn' in c else 0.4
    y = all_data[c]['val_loss']
    x = np.linspace(0, 15, num=len(y))
    if 'relu' in c:
      label = 'ReLU ' + label
      ax[0, 0].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 0].set_title('ReLU', fontsize=10)
    elif 'dsilu' in c:
      label = 'dSiLU ' + label
      ax[0, 2].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 2].set_title('DSiLU', fontsize=10)
    elif 'silu' in c:
      label = 'SiLU ' + label
      ax[0, 1].plot(x, y, color='black', label=label, alpha=alpha)
      ax[0, 1].set_title('SiLU', fontsize=10)

  for m in mnist_data:
    label = 'with BN' if 'bn' in m else 'without BN'
    alpha = 1.0 if 'bn' in m else 0.4
    y = all_data[m]['val_loss']
    x = np.linspace(0, 15, num=len(y))
    if 'relu' in m:
      label = 'ReLU ' + label
      ax[1, 0].plot(x, y, color='black', label=label, alpha=alpha)
    elif 'dsilu' in m:
      ax[1, 2].plot(x, y, color='black', label=label, alpha=alpha)
    elif 'silu' in m:
      ax[1, 1].plot(x, y, color='black', label=label, alpha=alpha)

  plt.legend(loc='right')

  ax[0, 0].set_ylabel('CIFAR-10', fontsize=10)
  ax[1, 0].set_ylabel('MNIST', fontsize=10)
  plt.show()


def plot_activations():
  x1 = np.linspace(-6, 6, 100)
  s = silu(x1)
  r = relu(x1)

  x2 = np.linspace(-8, 8, 100)
  ds = dsilu(x2)
  sg = sigmoid(x2)

  fig, ax = plt.subplots(2, 1, figsize=(5, 5))
  ax[0].plot(x1, r, label='ReLU', color='red')
  ax[0].plot(x1, s, label='SiLU', color='black')
  ax[0].set_xlim(left=x1[0], right=x1[-1])
  ax[0].set_ylim(top=np.max(s))
  ax[0].grid()
  ax[0].legend(loc='upper left')

  ax[1].plot(x2, ds, label='dSiLU', color='green')
  ax[1].plot(x2, sg, label='Sigmoid', color='orange')
  ax[1].set_xlim(left=x2[0], right=x2[-1])
  ax[1].grid()
  ax[1].legend(loc='upper left')

  # plt.legend(loc='upper left')
  plt.tight_layout()
  # plt.xlim(left=x[0], right=x[-1])
  plt.show()


if __name__ == '__main__':
    plot_activations()


