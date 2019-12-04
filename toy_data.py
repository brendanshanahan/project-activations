from architectures.fully_connected_models import *
from tensorflow.keras.layers import ReLU
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

TOY_DATA_SIZE = 256
TOY_DATA_VAL_SIZE = 2000
TRIALS = 25
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.01
SIGMA = 3.0

L1_UNITS = 100
L2_UNITS = 100
L3_UNITS = 100
L4_UNITS = 100


def target_function(x, epsilon1=0.0, epsilon2=0.0):
	return 0.3 * (np.sin(2 * np.pi * (x + epsilon1)) + np.sin(4 * np.pi * (x + epsilon1))) + epsilon2


def target_cubic(x, sigma=3.0):
	return x ** 3 + np.random.normal(0, sigma ** 2, x.shape[0])


def run_toy_data_trials(batch_norm=False, save_results=False, plot_results=True):
	x = np.linspace(0, 0.5, TOY_DATA_SIZE)
	y = target_function(x)
	eps1 = np.random.normal(0, 0.02, x.shape[0])
	eps2 = np.random.normal(0, 0.04, x.shape[0])
	y_perturb = target_function(x, eps1, eps2)
	x_val = np.linspace(-0.5, 1.0, 2000)
	y_val = target_function(x_val)

	if plot_results:
		plt.scatter(x, y_perturb, 2, 'k')
		plt.plot(x_val, y_val)
		plt.show()

	relu_pred = np.zeros(shape=(TRIALS, TOY_DATA_VAL_SIZE))
	silu_pred = np.zeros_like(relu_pred)
	dsilu_pred = np.zeros_like(relu_pred)

	for i in tqdm(range(TRIALS)):
		relu_model = FunctionApprox(layer1_units=L1_UNITS,
																layer2_units=L2_UNITS,
																layer3_units=L3_UNITS,
																layer4_units=L4_UNITS,
																activation=ReLU,
																batch_norm=batch_norm)

		silu_model = FunctionApprox(layer1_units=L1_UNITS,
																layer2_units=L2_UNITS,
																layer3_units=L3_UNITS,
																layer4_units=L4_UNITS,
																activation=SiLULayer,
																batch_norm=batch_norm)

		dsilu_model = FunctionApprox(layer1_units=L1_UNITS,
																 layer2_units=L2_UNITS,
																 layer3_units=L3_UNITS,
																 layer4_units=L4_UNITS,
																 activation=DSiLULayer,
																 batch_norm=batch_norm)

		x_train, x_test, y_train, y_test = train_test_split(x, y_perturb, test_size=0.25)

		lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
		optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

		for model in [relu_model, silu_model, dsilu_model]:
			model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

		relu_history = relu_model.fit(x_train,
																	y_train,
																	batch_size=BATCH_SIZE,
																	epochs=EPOCHS,
																	validation_data=(x_test, y_test),
																	callbacks=[lr_callback],
																	# verbose=0,
																	)

		silu_history = silu_model.fit(x_train,
																	y_train,
																	batch_size=BATCH_SIZE,
																	epochs=EPOCHS,
																	validation_data=(x_test, y_test),
																	callbacks=[lr_callback],
																	# verbose=0,
																	)

		dsilu_history = dsilu_model.fit(x_train,
																		y_train,
																		batch_size=BATCH_SIZE,
																		epochs=EPOCHS,
																		validation_data=(x_test, y_test),
																		callbacks=[lr_callback],
																		# verbose=0,
																		)

		relu_pred[i] = relu_model.predict(x_val).reshape(-1,)
		silu_pred[i] = silu_model.predict(x_val).reshape(-1,)
		dsilu_pred[i] = dsilu_model.predict(x_val).reshape(-1,)

	if save_results:
		save_data = {'relu': relu_pred,
								 'silu': silu_pred,
								 'dsilu': dsilu_pred
								 }

		with open('toy-data.pickle', 'wb') as f:
			pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

	if plot_results:
		relu_mean = np.mean(relu_pred, axis=0)
		silu_mean = np.mean(silu_pred, axis=0)
		dsilu_mean = np.mean(dsilu_pred, axis=0)

		relu_std = np.std(relu_pred, axis=0)
		silu_std = np.std(silu_pred, axis=0)
		dsilu_std = np.std(dsilu_pred, axis=0)

		fig, ax = plt.subplots(1, 3)
		[ax[i].plot(x_val, y_val, label='true', color='orange') for i in range(3)]
		[ax[i].scatter(x, y_perturb, 2, 'k') for i in range(3)]

		ax[0].fill_between(x_val, relu_mean + 2*relu_std, relu_mean - 2*relu_std, alpha=0.5, color='purple')
		ax[0].fill_between(x_val, relu_mean + relu_std, relu_mean - relu_std, alpha=0.5, color='blue')
		ax[0].plot(x_val, relu_mean, label='relu', color='red')
		ax[0].set_title('ReLU')
		ax[1].fill_between(x_val, silu_mean + 2*silu_std, silu_mean - 2*silu_std, alpha=0.5, color='purple')
		ax[1].fill_between(x_val, silu_mean + silu_std, silu_mean - silu_std, alpha=0.5, color='blue')
		ax[1].plot(x_val, silu_mean, label='silu', color='red')
		ax[1].set_title('SiLU')
		ax[2].fill_between(x_val, dsilu_mean + 2*dsilu_std, dsilu_mean - 2*dsilu_std, alpha=0.5, color='purple')
		ax[2].fill_between(x_val, dsilu_mean + dsilu_std, dsilu_mean - dsilu_std, alpha=0.5, color='blue')
		ax[2].plot(x_val, dsilu_mean, label='dsilu', color='red')
		ax[2].set_title('DSiLU')

		plt.show()


if __name__ == '__main__':
	run_toy_data_trials(batch_norm=True)

