from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline
import json
import time
from sequential_model import sequential_model
import os

# initialize data pipeline and parameters
pipe = MfccPipeline()
num_files = 512
num_batches = 100
data_generator = pipe.get_dataset(num_files=num_files, num_batches=num_batches)

# initialize model and parameters
batch_size = 128
epochs = 10
model_init = sequential_model(name = "sequential_1_50000samps")


model = model_init.get_model()

# delete the old performance log file if it exists
if os.path.exists("model_results/{}_performance_log.txt".format(model.name)):
	os.remove("model_results/{}_performance_log.txt".format(model.name))

csv_logger = keras.callbacks.CSVLogger('model_results/{}.log'.format(model.name), append=True)

# run model
total_start = time.time()

# train model
for batch in range(num_batches):
	load_start = time.time()
	data = next(data_generator)
	load_end = time.time()
	perf_str = "batch # {}; loading time: {}s\n".format(batch, (load_end - load_start))
	print(perf_str)
	with open("model_results/{}_performance_log.txt".format(model.name), 'a') as f:
		f.write(perf_str)

	X = data[0]
	t = data[1]

	# if this is the last batch, use it to evaluate the model
	if batch == num_batches-1:
		print("evaluating")
		results = model.evaluate(X, t, batch_size=batch_size, callbacks=[csv_logger])
	else:
		model.fit(X, t, batch_size=batch_size, shuffle=True, epochs=epochs, callbacks=[csv_logger])

total_end = time.time()
total_str = "total time elapsed: {}\n".format((total_end-total_start))
print(total_str)
with open("model_results/{}_performance_log.txt".format(model.name), 'a') as f:
	f.write(total_str)
	f.write("validation loss: {}, validation accuracy: {}".format(results[0], results[1]))

# save the model
model.save('model_results/{}_saved'.format(model.name))
