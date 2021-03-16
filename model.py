from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline

#### Try running a basic sequential model

# prepare the data
pipe = MfccPipeline()
X_train, X_val, y_train, y_val = pipe.mfcc_pipeline(num_samples = 100000)

# build the network architecture
model = keras.Sequential([
    # input layer                    # flatten this 2D array, intervals x values of MFCCs for that interval
    keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),

    # 1st hidden layer
    keras.layers.Dense(1024, activation="relu"),

    # 2nd hidden layer
    keras.layers.Dense(512, activation="relu"),

    # 3rd hidden layer
    keras.layers.Dense(256, activation="relu"),

    # 4th hidden layer
    keras.layers.Dense(64,  activation="relu"),

    # 5th hidden layer
    keras.layers.Dense(y_train.shape[1], activation='relu'),

    # output layers- 1 neuron for each genre
    keras.layers.Dense(y_train.shape[1], activation="softmax")
])

# compile network
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
model.summary()

#model.fit(X_train, y_train)
history = model.fit(X_train, y_train, batch_size=64, shuffle=True, validation_data=(X_val, y_val), epochs=5)

with open("history_basic.txt", 'w') as f:
    f.write(history)