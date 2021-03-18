from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline
import json
import time


class sequential_model:

	def __init__(self, name="basic_sequential"):
		self.name = name
		# build the network architecture
		self.model = keras.Sequential([
			# input layer                    # flatten this 2D array, intervals x values of MFCCs for that interval
			keras.layers.Flatten(input_shape=(20, 126)),

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
		optimizer = keras.optimizers.Adam(learning_rate=0.01)
		self.model.compile(optimizer=optimizer,
					loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
					metrics=["accuracy"])
		self.model.summary()

	def get_model(self):
		return self.model

