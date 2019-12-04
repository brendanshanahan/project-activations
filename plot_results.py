import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import json

# global plotting parameters and data file locations
SMOOTHING = 0.75
ALPHA = 0.3
LINEWIDTH = 1
SMOOTH_LINEWIDTH = 1

mpl.rcParams['lines.linewidth'] = 1

# toy data
TOY_DATA_RESULTS = 'results/toy-data.pickle'
TOY_DATA_RESULTS_BN = 'results/toy-data-bn.pickle'

TOY_DATA_SIZE = 256
TOY_DATA_VAL_SIZE = 10000

# cifar10
CIFAR10_ALL_RESULTS_PATH = 'results/cifar10-all-trials.pickle'

# fitnet
FITNET_RELU_PATH = 'results/fitnet-relu-cifar100.pickle'
FITNET_SILU_PATH = 'results/fitnet-silu-cifar100.pickle'
FITNET_DSILU_PATH = 'results/fitnet-dsilu-cifar100.pickle'

# resnet
RESNET_RELU_ACC_PATH = 'results/resnet-relu-cifar100-acc.json'
RESNET_SILU_ACC_PATH = 'results/resnet-silu-cifar100-acc.json'
RESNET_DSILU_ACC_PATH = 'results/resnet-dsilu-cifar100-acc.json'

RESNET_RELU_LOSS_PATH = 'results/resnet-relu-cifar100-loss.json'
RESNET_SILU_LOSS_PATH = 'results/resnet-silu-cifar100-loss.json'
RESTNET_DSILU_LOSS_PATH = 'results/resnet-dsilu-cifar100-loss.json'


def relu(x):
	return np.where(x > 0, x, 0)


def sigmoid(x):
	return 1/(1 + np.exp(-x))


def silu(x):
	return x * sigmoid(x)


def dsilu(x):
	s = sigmoid(x)
	return s * (1 + x*(1 - s))


def smooth(x, weight=0.0):
	"""
	exponential smoothing for noisy data
	"""
	last = x[0]
	smoothed = []
	for point in x:
		smoothed_val = last * weight + (1 - weight) * point
		smoothed.append(smoothed_val)
		last = smoothed_val
	return smoothed


def plot_cifar10_results():
	mpl.rcParams['lines.linewidth'] = 1.25

	with open(CIFAR10_ALL_RESULTS_PATH, 'rb') as f:
		all_data = pickle.load(f)

	cifar10_data = [k for k in all_data.keys() if 'cifar10' in k]
	mnist_data = [k for k in all_data.keys() if 'mnist' in k]

	fig, ax = plt.subplots(2, 2, figsize=(6, 4), sharex='col')
	plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.92, wspace=0.18, hspace=0.15)

	for c in cifar10_data:
		bn = ('bn' in c)
		label = '(w/ BN)' if bn else ''
		alpha = 1.0 if bn else 0.6
		acc = all_data[c]['val_sparse_categorical_accuracy']
		loss = all_data[c]['val_loss']
		x = np.linspace(0, 15, num=len(acc))
		if bn and 'relu' in c:
			label = 'ReLU ' + label
			ax[0, 0].plot(x, acc, color='black', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='black', label=label, alpha=alpha)
		elif 'relu' in c:
			label = 'ReLU ' + label
			ax[0, 0].plot(x, acc, color='black', ls='--', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='black', ls='--', label=label, alpha=alpha)
		elif bn and 'dsilu' in c:
			label = 'DSiLU ' + label
			ax[0, 0].plot(x, acc, color='green', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='green', label=label, alpha=alpha)
		elif 'dsilu' in c:
			label = 'dSiLU ' + label
			ax[0, 0].plot(x, acc, color='green', ls='--', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='green', ls='--', label=label, alpha=alpha)
		elif bn and 'silu' in c:
			label = 'SiLU ' + label
			ax[0, 0].plot(x, acc, color='red', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='red', label=label, alpha=alpha)
		elif 'silu' in c:
			label = 'SiLU ' + label
			ax[0, 0].plot(x, acc, color='red', ls='--', label=label, alpha=alpha)
			ax[1, 0].plot(x, loss, color='red', ls='--', label=label, alpha=alpha)

	for m in mnist_data:
		bn = ('bn' in m)
		label = '(w/ BN)' if bn else ''
		alpha = 1.0 if bn else 0.6
		acc = all_data[m]['val_sparse_categorical_accuracy']
		loss = all_data[m]['val_loss']
		x = np.linspace(0, 15, num=len(acc))

		if bn and 'relu' in m:
			label = 'ReLU ' + label
			ax[0, 1].plot(x, acc, color='black', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='black', label=label, alpha=alpha)
		elif 'relu' in m:
			label = 'ReLU ' + label
			ax[0, 1].plot(x, acc, color='black', ls='--', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='black', ls='--', label=label, alpha=alpha)
		elif bn and 'dsilu' in m:
			label = 'DSiLU ' + label
			ax[0, 1].plot(x, acc, color='green', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='green', label=label, alpha=alpha)
		elif 'dsilu' in m:
			label = 'DSiLU ' + label
			ax[0, 1].plot(x, acc, color='green', ls='--', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='green', ls='--', label=label, alpha=alpha)
		elif bn and 'silu' in m:
			label = 'SiLU ' + label
			ax[0, 1].plot(x, acc, color='red', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='red', label=label, alpha=alpha)
		elif 'silu' in m:
			label = 'SiLU ' + label
			ax[0, 1].plot(x, acc, color='red', ls='--', label=label, alpha=alpha)
			ax[1, 1].plot(x, loss, color='red', ls='--', label=label, alpha=alpha)

	plt.legend(loc='upper right', fontsize=8)
	ax[0, 0].set_title('CIFAR-10', fontsize=10)
	ax[0, 1].set_title('MNIST', fontsize=10)
	plt.show()


def plot_cifar10_val_accuracy():
	with open(CIFAR10_ALL_RESULTS_PATH, 'rb') as f:
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


def plot_cifar10_val_loss():
	with open(CIFAR10_ALL_RESULTS_PATH, 'rb') as f:
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


def plot_fitnet_cifar100_results(with_error_bars=False):
	acc = 'val_sparse_categorical_accuracy'
	loss = 'val_loss'

	epochs = 75
	samples = 10

	relu_acc_sim = np.zeros(shape=(samples, epochs))
	silu_acc_sim = np.zeros(shape=(samples, epochs))
	dsilu_acc_sim = np.zeros(shape=(samples, epochs))

	relu_loss_sim = np.zeros(shape=(samples, epochs))
	silu_loss_sim = np.zeros(shape=(samples, epochs))
	dsilu_loss_sim = np.zeros(shape=(samples, epochs))

	with open(FITNET_RELU_PATH, 'rb') as f:
		relu_data = pickle.load(f)

	with open(FITNET_SILU_PATH, 'rb') as f:
		silu_data = pickle.load(f)

	with open(FITNET_DSILU_PATH, 'rb') as f:
		dsilu_data = pickle.load(f)

	i = 0
	for r, s, d in zip(relu_data.values(), silu_data.values(), dsilu_data.values()):
		r_acc, _ = r
		s_acc, _ = s
		d_acc, _ = d

		relu_acc_sim[i], relu_loss_sim[i] = r_acc[acc], r_acc[loss]
		silu_acc_sim[i], dsilu_loss_sim[i] = s_acc[acc], s_acc[loss]
		dsilu_acc_sim[i], silu_loss_sim[i] = d_acc[acc], d_acc[loss]

		i += 1

	relu_acc_mean = np.mean(relu_acc_sim, axis=0)
	silu_acc_mean = np.mean(silu_acc_sim, axis=0)
	dsilu_acc_mean = np.mean(dsilu_acc_sim, axis=0)

	relu_loss_mean = np.mean(relu_loss_sim, axis=0)
	silu_loss_mean = np.mean(silu_loss_sim, axis=0)
	dsilu_loss_mean = np.mean(dsilu_loss_sim, axis=0)
	x = np.linspace(1, epochs, epochs)

	fig, ax = plt.subplots(2, 1, figsize=(6, 7), sharex='col')

	if not with_error_bars:
		ax[0].plot(relu_acc_mean, color='k', label='ReLU (w/ variance)')
		ax[0].plot(silu_acc_mean, color='r', label='SiLU (w/ variance)')
		ax[0].plot(dsilu_acc_mean, color='g', label='DSiLU (w/ variance)')
		ax[0].set_ylabel('Validation Accuracy')

		ax[1].plot(relu_loss_mean, color='k', label='ReLU')
		ax[1].plot(silu_loss_mean, color='r', label='SiLU')
		ax[1].plot(dsilu_loss_mean, color='g', label='DSiLU')
		ax[1].set_xlabel('Epoch')
		ax[1].set_ylabel('Validation Loss')
		ax[1].legend()

	else:
		relu_loss_var = np.var(relu_loss_sim, axis=0)
		silu_loss_var = np.var(silu_loss_sim, axis=0)
		dsilu_loss_var = np.var(dsilu_loss_sim, axis=0)

		relu_acc_var = np.var(relu_acc_sim, axis=0)
		silu_acc_var = np.var(silu_acc_sim, axis=0)
		dsilu_acc_var = np.var(dsilu_acc_sim, axis=0)

		ax[0].errorbar(x=x, y=relu_acc_mean, yerr=relu_acc_var, color='k', abel='ReLU (w/ variance)')
		ax[0].errorbar(x=x, y=silu_acc_mean, yerr=silu_acc_var, color='r', label='SiLU (w/ variance)')
		ax[0].errorbar(x=x, y=dsilu_acc_mean, yerr=dsilu_acc_var, color='g', label='DSiLU (w/ variance)')
		ax[0].set_ylabel('Validation Accuracy')
		ax[0].legend()

		ax[1].errorbar(x=x+0.5, y=relu_loss_mean, yerr=relu_loss_var, color='k', label='ReLU (variance)')
		ax[1].errorbar(x=x-0.5, y=silu_loss_mean, yerr=silu_loss_var, color='r', label='SiLU (variance)')
		ax[1].errorbar(x=x, y=dsilu_loss_mean, yerr=dsilu_loss_var, color='g', label='DSiLU (variance)')
		ax[1].set_xlabel('Epoch')
		ax[1].set_ylabel('Validation Loss')

	plt.subplots_adjust(left=0.12, bottom=0.09, right=0.95, top=0.97, hspace=0.13)
	plt.show()


def plot_resnet_cifar100_results():
	with open(RESNET_RELU_ACC_PATH, 'r') as f:
		relu_acc = np.array(json.load(f))

	with open(RESNET_SILU_ACC_PATH, 'r') as f:
		silu_acc = np.array(json.load(f))

	with open(RESNET_DSILU_ACC_PATH, 'r') as f:
		dsilu_acc = np.array(json.load(f))

	with open(RESNET_RELU_LOSS_PATH, 'r') as f:
		relu_loss = np.array(json.load(f))

	with open(RESNET_SILU_LOSS_PATH, 'r') as f:
		silu_loss = np.array(json.load(f))

	with open(RESTNET_DSILU_LOSS_PATH, 'r') as f:
		dsilu_loss = np.array(json.load(f))

	fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex='col')

	ax[0].set_title('Validation Accuracy')
	ax[0].plot(relu_acc[:, 1], relu_acc[:, 2], color='k', alpha=ALPHA)
	ax[0].plot(relu_acc[:, 1], smooth(relu_acc[:, 2], SMOOTHING), label='ReLU', color='k')
	ax[0].plot(silu_acc[:, 1], silu_acc[:, 2], color='r', alpha=ALPHA)
	ax[0].plot(silu_acc[:, 1], smooth(silu_acc[:, 2], SMOOTHING), label='SiLU', color='r')
	ax[0].plot(dsilu_acc[:, 1], dsilu_acc[:, 2], color='g', alpha=ALPHA)
	ax[0].plot(dsilu_acc[:, 1], smooth(dsilu_acc[:, 2], SMOOTHING), label='DSiLU', color='g')

	ax[1].set_title('Validation Loss')
	ax[1].plot(relu_loss[:, 1], relu_loss[:, 2], color='k', alpha=ALPHA)
	ax[1].plot(relu_loss[:, 1], smooth(relu_loss[:, 2]), label='ReLU', color='k')
	ax[1].plot(silu_loss[:, 1], silu_loss[:, 2], color='r', alpha=ALPHA)
	ax[1].plot(silu_loss[:, 1], smooth(silu_loss[:, 2], SMOOTHING), label='SiLU', color='r')
	ax[1].plot(dsilu_loss[:, 1], dsilu_loss[:, 2], color='g', alpha=ALPHA)
	ax[1].plot(dsilu_loss[:, 1], smooth(dsilu_loss[:, 2], SMOOTHING), label='DSiLU', color='g')
	ax[1].legend()

	plt.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.94)
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

	plt.tight_layout()
	plt.show()


# simulate toy data
def target_function(x, eps=0.0):
	return 0.3 * (np.sin(2 * np.pi * (x + eps)) + np.sin(4 * np.pi * (x + eps))) + eps


def plot_toy_data_results():
	with open(TOY_DATA_RESULTS_BN, 'rb') as f:
		bn_data = pickle.load(f)

	with open(TOY_DATA_RESULTS, 'rb') as f:
		reg_data = pickle.load(f)

	relu_mean = np.mean(reg_data['relu'], axis=0)
	silu_mean = np.mean(reg_data['silu'], axis=0)
	dsilu_mean = np.mean(reg_data['dsilu'], axis=0)

	relu_std = np.std(reg_data['relu'], axis=0)
	silu_std = np.std(reg_data['silu'], axis=0)
	dsilu_std = np.std(reg_data['dsilu'], axis=0)

	relu_mean_bn = np.mean(bn_data['relu'], axis=0)
	silu_mean_bn = np.mean(bn_data['silu'], axis=0)
	dsilu_mean_bn = np.mean(bn_data['dsilu'], axis=0)

	relu_std_bn = np.std(bn_data['relu'], axis=0)
	silu_std_bn = np.std(bn_data['silu'], axis=0)
	dsilu_std_bn = np.std(bn_data['dsilu'], axis=0)

	# funky fn
	x = np.linspace(0, 0.5, TOY_DATA_SIZE)
	eps = np.random.normal(0, 0.04, x.shape[0])
	y_perturb = target_function(x, eps)
	x_val = np.linspace(-0.5, 1.0, TOY_DATA_VAL_SIZE)
	y_val = target_function(x_val)

	fig, ax = plt.subplots(2, 3, figsize=(8, 6), sharex='col', sharey='row')
	[ax[i, j].plot(x_val, y_val, label='Ground Truth', color='orange', linewidth=1.5) for i in range(2) for j in range(3)]
	[ax[i, j].scatter(x, y_perturb, 1, 'k', alpha=0.7) for i in range(2) for j in range(3)]

	# no batch norm
	ax[0, 0].fill_between(x_val, relu_mean + 2 * relu_std, relu_mean - 2 * relu_std, alpha=0.5, color='purple')
	ax[0, 0].fill_between(x_val, relu_mean + relu_std, relu_mean - relu_std, alpha=0.5, color='blue')
	ax[0, 0].plot(x_val, relu_mean, label='relu', color='red', linewidth=1.5)
	ax[0, 0].set_title('ReLU')
	ax[0, 0].set_ylabel('No Batch Normalization')
	ax[0, 1].fill_between(x_val, silu_mean + 2 * silu_std, silu_mean - 2 * silu_std, alpha=0.5, color='purple')
	ax[0, 1].fill_between(x_val, silu_mean + silu_std, silu_mean - silu_std, alpha=0.5, color='blue')
	ax[0, 1].plot(x_val, silu_mean, label='silu', color='red', linewidth=1.5)
	ax[0, 1].set_title('SiLU')
	ax[0, 2].fill_between(x_val, dsilu_mean + 2 * dsilu_std, dsilu_mean - 2 * dsilu_std, alpha=0.5, color='purple')
	ax[0, 2].fill_between(x_val, dsilu_mean + dsilu_std, dsilu_mean - dsilu_std, alpha=0.5, color='blue')
	ax[0, 2].plot(x_val, dsilu_mean, label='dsilu', color='red', linewidth=1.5)
	ax[0, 2].set_title('DSiLU')

	# batch norm
	ax[1, 0].fill_between(x_val, relu_mean_bn + 2 * relu_std_bn, relu_mean_bn - 2 * relu_std_bn, alpha=0.5,
												color='purple')
	ax[1, 0].fill_between(x_val, relu_mean_bn + relu_std_bn, relu_mean_bn - relu_std_bn, alpha=0.5, color='blue')
	ax[1, 0].plot(x_val, relu_mean_bn, label='ReLU', color='red', linewidth=1.5)
	ax[1, 0].set_ylabel('Batch Normalization')
	ax[1, 1].fill_between(x_val, silu_mean_bn + 2 * silu_std_bn, silu_mean_bn - 2 * silu_std_bn, alpha=0.5,
												color='purple')
	ax[1, 1].fill_between(x_val, silu_mean_bn + silu_std_bn, silu_mean_bn - silu_std_bn, alpha=0.5, color='blue')
	ax[1, 1].plot(x_val, silu_mean_bn, label='SiLU', color='red', linewidth=1.5)
	ax[1, 2].fill_between(x_val, dsilu_mean_bn + 2 * dsilu_std_bn, dsilu_mean_bn - 2 * dsilu_std_bn, alpha=0.5,
												color='purple')
	ax[1, 2].fill_between(x_val, dsilu_mean_bn + dsilu_std_bn, dsilu_mean_bn - dsilu_std_bn, alpha=0.5, color='blue')
	ax[1, 2].plot(x_val, dsilu_mean_bn, label='Mean Prediction', color='red', linewidth=1.5)
	[ax[i, j].set_xlim(-0.1, 0.6) for i in range(2) for j in range(3)]
	[ax[i, j].set_ylim(-1.0, 1.0) for i in range(2) for j in range(3)]

	plt.subplots_adjust(left=0.11, bottom=0.06, right=0.95, top=0.95, hspace=0.08, wspace=0.09)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	plot_cifar10_results()
	plot_fitnet_cifar100_results()
	plot_resnet_cifar100_results()
	plot_toy_data_results()
