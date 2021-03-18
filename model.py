from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline
import json
import time

#### Try running a basic sequential model

# initialize model and data pipeline
pipe = MfccPipeline()

# build the network architecture
model = keras.Sequential([
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
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
model.summary()

# loop through and train model
num_files = 512
num_batches = 10
data_generator = pipe.get_dataset(num_files=num_files, num_batches=num_batches)
csv_logger = keras.callbacks.CSVLogger('training.log', append=True)

total_start = time.time()
for batch in range(num_batches):

    load_start = time.time()
    data = next(data_generator)
    load_end = time.time()
    print("batch # {}; loading time: {}s".format(batch, (load_end - load_start)))

    X = data[0]
    t = data[1]
    model.fit(X, t, batch_size=64, shuffle=True, epochs=10, callbacks=[csv_logger])
total_end = time.time()
print("total time elapsed: ", (end-start))
# save the model
