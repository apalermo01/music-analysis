from tensorflow import keras
import tensorflow as tf
from train_data_preparation import MfccPipeline
import json
import time
import models
import os
import numpy as np
import pandas as pd

def train_a_model(train_path="C:/Users/alexc/Downloads/nsynth-train.jsonwav.tar.gz",
					num_files=512, num_batches=10,
					batch_size=128, epochs=10,
					model_init=models.sequential_model("Generic name"),
					save_model=False, save_perf=False,
					out_path = "model_results/"):

	# initialize model and parameters
	pipe = MfccPipeline()
	data_generator = pipe.get_dataset(num_files=num_files, num_batches=num_batches)
	model = model_init.get_model()

	# delete the old performance log file if it exists
	if os.path.exists("{}/{}_performance_log.txt".format(out_path, model_init.name)) and save_perf:
		os.remove("{}/{}_performance_log.txt".format(out_path, model_init.name))

	csv_logger = keras.callbacks.CSVLogger('{}/{}.log'.format(out_path, model_init.name), append=True)

	# run model
	total_start = time.time()

	# train model
	for batch in range(num_batches):
		load_start = time.time()
		data = next(data_generator)
		load_end = time.time()
		perf_str = "batch # {}; loading time: {}s\n".format(batch, (load_end - load_start))
		print(perf_str)
		if save_perf:
			with open("{}/{}_performance_log.txt".format(out_path, model_init.name), 'a') as f:
				f.write(perf_str)

		X = data[0]
		t = data[1]

		### further modification for cnn
		if "cnn" in model_init.name:
			X = X[..., np.newaxis]

		# if this is the last batch, use it to evaluate the model
		if batch == num_batches-1:
			print("evaluating")
			results = model.evaluate(X, t, batch_size=batch_size, callbacks=[csv_logger])
		else:
			model.fit(X, t, batch_size=batch_size, shuffle=True, epochs=epochs, callbacks=[csv_logger])

	total_end = time.time()
	total_str = "total time elapsed: {}\n".format((total_end-total_start))
	print(total_str)
	if save_perf:
		with open("{}/{}_performance_log.txt".format(out_path, model_init.name), 'a') as f:
			f.write(total_str)
			f.write("validation loss: {}, validation accuracy: {}".format(results[0], results[1]))

	# save the model
	if save_model:
		model.save('{}/{}_saved'.format(out_path, model_init.name))

	# return the results
	return results, model_init.hyper_params

if __name__ == '__main__':

	args={
		'train_path': "C:/Users/alexc/Downloads/nsynth-train.jsonwav.tar.gz",
		'num_files': 1024,
		'num_batches': 100,
		'batch_size': 256,
		'epochs': 10,
		#'model_init': models.sequential_model(name = "DELETE THIS MODEL- DEBUGGING ONLY"),
		'save_model': True,
		'save_perf': True,
		'out_path': "model_results/batch_3182021",
	}

	models = [
		models.sequential_model(name="sequential_lr_0.1", lr=0.1),
		models.sequential_model(name="sequential_lr_0.01", lr=0.01),
		models.sequential_model(name="sequential_lr_0.001", lr=0.001),
		models.sequential_model(name="sequential_lr_0.0001", lr=0.0001),
		models.sequential_model_short(name="short_sequential_lr_0.1", lr=0.1),
		models.sequential_model_short(name="short_sequential_lr_0.01", lr=0.01),
		models.sequential_model_short(name="short_sequential_lr_0.001", lr=0.001),
		models.sequential_model_short(name="short_sequential_lr_0.0001", lr=0.0001),
		models.sequential_l2(name="sequential_lr_0.01_l2_0.00001", lr=0.01, l2_reg=0.00001),
		models.sequential_l2(name="sequential_lr_0.01_l2_0.001", lr=0.01, l2_reg=0.001),
		models.sequential_l2(name="sequential_lr_0.001_l2_0.00001", lr=0.001, l2_reg=0.00001),
		models.sequential_l2(name="sequential_lr_0.001_l2_0.001", lr=0.001, l2_reg=0.001),
		#models.cnn_basic(name="test_cnn_basic"),
		#models.rnn_lstm_basic(name="test_rnn_lstm_basic"),
	]

	results_df = pd.DataFrame()
	for model in models:
		print("training: ", model.name)
		results, hyper_params = train_a_model(**args, model_init=model)
		result_dict = dict({'name': model.name}, **model.hyper_params)
		result_dict.update({'loss': results[0], 'accuracy': results[1]})

		results_df = results_df.append(result_dict, ignore_index=True)

	print(results_df.head)
	results_df.to_csv("model_results/batch_3182021/results_3182021.csv")