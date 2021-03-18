from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline
import json
import time


class sequential_model:

	def __init__(self, name="basic_sequential",
				input_shape_x=20, input_shape_y=126, lr=0.01):

		self.name = name
		self.hyper_params = {'Learning rate': lr}

		# build the network architecture
		self.model = keras.Sequential([
			# input layer                    # flatten this 2D array, intervals x values of MFCCs for that interval
			keras.layers.Flatten(input_shape=(input_shape_x, input_shape_y)),

			# 1st hidden layer
			keras.layers.Dense(2048, activation="relu"),
			keras.layers.Dropout(0.3),

			# 1.5 hidden layer
			keras.layers.Dense(1024, activation="relu"),
			keras.layers.Dropout(0.3),

			# 2nd hidden layer
			keras.layers.Dense(512, activation="relu"),
			keras.layers.Dropout(0.3),

			# 3rd hidden layer
			keras.layers.Dense(256, activation="relu"),
			keras.layers.Dropout(0.3),

			# 4th hidden layer
			keras.layers.Dense(64,  activation="relu"),
			keras.layers.Dropout(0.3),

			# 5th hidden layer
			keras.layers.Dense(11, activation='relu'),

			# output layers- 1 neuron for each genre
			keras.layers.Dense(11, activation="softmax")
		])

		# compile network
		optimizer = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
					metrics=["accuracy"])
		self.model.summary()

	def get_model(self):
		return self.model

class sequential_model_short:

	def __init__(self, name="sequential_short",
				input_shape_x=20, input_shape_y=126, lr=0.01):

		self.name = name
		self.hyper_params = {'Learning rate': lr}
		# build the network architecture
		self.model = keras.Sequential([
			# input layer                    # flatten this 2D array, intervals x values of MFCCs for that interval
			keras.layers.Flatten(input_shape=(input_shape_x, input_shape_y)),

			# 1st hidden layer
			keras.layers.Dense(512, activation="relu"),
			keras.layers.Dropout(0.3),

			# 2nd hidden layer
			keras.layers.Dense(64,  activation="relu"),
			keras.layers.Dropout(0.3),

			# # 5th hidden layer
			# keras.layers.Dense(11, activation='relu'),

			# output layers- 1 neuron for each genre
			keras.layers.Dense(11, activation="softmax")
		])

		# compile network
		optimizer = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
					metrics=["accuracy"])
		self.model.summary()

	def get_model(self):
		return self.model

class sequential_l2:

	def __init__(self, name="sequential_l2",
				input_shape_x=20, input_shape_y=126, lr=0.01,
				l2_reg=0.001):

		self.name = name
		self.hyper_params = {'Learning rate': lr, 'l2 regularization': l2_reg}

		# build the network architecture
		self.model = keras.Sequential([
			# input layer                    # flatten this 2D array, intervals x values of MFCCs for that interval
			keras.layers.Flatten(input_shape=(input_shape_x, input_shape_y)),

			# 1st hidden layer
			keras.layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)),
			keras.layers.Dropout(0.3),

			# 1.5 hidden layer
			keras.layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)),
			keras.layers.Dropout(0.3),

			# 2nd hidden layer
			keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)),
			keras.layers.Dropout(0.3),

			# 3rd hidden layer
			keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)),
			keras.layers.Dropout(0.3),

			# 4th hidden layer
			keras.layers.Dense(64,  activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)),
			keras.layers.Dropout(0.3),

			# 5th hidden layer
			keras.layers.Dense(11, activation='relu'),

			# output layers- 1 neuron for each genre
			keras.layers.Dense(11, activation="softmax")
		])

		# compile network
		optimizer = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
					metrics=["accuracy"])
		self.model.summary()

	def get_model(self):
		return self.model

class cnn_basic:
	# create model
	def __init__(self, name="cnn_basic",
				input_shape_x=20, input_shape_y=126, lr=0.001):

		self.name = name
		self.hyper_params = {'Learning rate': lr}

		self.model = keras.Sequential()

		# This network was copied directly from https://www.youtube.com/watch?v=dOG-HxpbMSw&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=16
		# use .add() to add a layer to the model
		# 1st conv layer
		self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape_x, input_shape_y, 1)))
		self.model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

		# The math behind BatchNormalization is pretty complicated (well beyond this scope)
		# intuition: standardizes / normalizes activations in current layer
		# speeds up the training (faster convergence)
		self.model.add(keras.layers.BatchNormalization())

		# 2nd conv layer
		self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape_x, input_shape_y, 1)))
		self.model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
		self.model.add(keras.layers.BatchNormalization())

		# 3rd conv layer
		self.model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(input_shape_x, input_shape_y, 1)))
		self.model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
		self.model.add(keras.layers.BatchNormalization())

		# flatten the output and feed it into dense layer
		self.model.add(keras.layers.Flatten())
		self.model.add(keras.layers.Dense(64, activation='relu'))

		# add dropout to avoid overfitting
		self.model.add(keras.layers.Dropout(0.3))

		# output layer (using softmax)
		self.model.add(keras.layers.Dense(11, activation="softmax"))

		optimizer = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=optimizer,
					loss="sparse_categorical_crossentropy",
					metrics=["accuracy"])
	def get_model(self):
		return self.model

class rnn_lstm_basic:

	def __init__(self, name="rnn-lstm_basic",
				input_shape_x=20, input_shape_y=126, lr=0.0001):

		self.name = name
		self.hyper_params = {'Learning rate': lr}

		# Create model
		self.model = keras.Sequential()

		# 2 LSTM layers
		# return_sequences: this is a sequence-to-sequence layer b/c we want to pass this to the second LSTM layer
		self.model.add(keras.layers.LSTM(64, input_shape=(input_shape_x, input_shape_y), return_sequences=True))

		# this one is a sequence-to-vector layer (default)
		self.model.add(keras.layers.LSTM(64))

		# dense layer
		self.model.add(keras.layers.Dense(64, activation='relu'))

		# dropout layer
		self.model.add(keras.layers.dropout(0.3))

		# output layer (using softmax)
		self.model.add(keras.layers.Dense(11, activation="softmax"))

		optimizer = keras.optimizers.Adam(learning_rate=lr)
		self.model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

	def get_model(self):
		return self.model