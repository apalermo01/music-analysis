import dataset
import numpy as np
from dataset import prep_dataset
from sklearn.metrics import precision_recall_fscore_support
import models
from models_dict import models_dict
import torch
from torchsummary import summary
import time
import matplotlib.pyplot as plt

"""Generic training loop"""


def train_model(filename="irmas_data_mfcc13_hop_length256_n_fft2048", model_id="TestModel",
								num_epochs=2, interval=16, lr=0.001, batch_size=64,
								val_split=0.2, save_checkpoint=False, checkpoint_path="",
								notes="", checkpoint_name="utitled.pt", criterion=torch.nn.NLLLoss(),
								patience=None, min_epochs=5, buffer=0.05, dropout_prob=None,
								model_args={}, experiment_params={}):
	"""Model training loop for music analysis project. Currently, this loop only supports
	models that take input in the shape [mini_batch, channels, L, W].

	:param filename:
	:param model_id:
	:param num_epochs:
	:param interval:
	:param lr:
	:param batch_size:
	:param val_split:
	:param save_checkpoint:
	:param checkpoint_path:
	:param notes:
	:param checkpoint name:
	:param criterion:
	:param patience: If validation loss does not improve over this many epochs, stop training
	"""

	# Initialize device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("device: ", device)
	
	# get train and validation set, print metadata
	train_loader, val_loader, dataset = prep_dataset(
			filename=filename, batch_size=batch_size, val_split=val_split)

	print("dataset metadata: ", dataset.metadata)

	# get number of train and validdation samples
	train_samples = round(len(dataset) * (1-val_split))
	val_samples = round(len(dataset)*val_split)

	# initialize loss history and accuracy history for each epoch
	# this is the stored history for the train and validation metrics
	epoch_hist = []
	avg_train_loss_hist = []  # training loss for each epoch
	std_train_loss_hist = []
	avg_val_loss_hist = []    # validation loss for each epoch
	std_val_loss_hist = []
	train_acc_hist = []       # training accuracy for each epoch
	train_prec_hist = []
	train_recall_hist = []
	train_f1_hist = []
	val_acc_hist = []         # validation accuracy for each epoch
	val_prec_hist = []
	val_recall_hist = []
	val_f1_hist = []


	# get one sample to load initial shape for neural net
	single_sample = dataset[0]
	one_mfcc = np.array(single_sample['mfccs'])
	print("train model: data loaders initialized")
	print("sample shape = ", one_mfcc.shape)

	# initialize model
	#model = models_dict[model_id](one_mfcc, dropout_prob).to(device)
	model = models_dict[model_id](one_mfcc, **model_args).to(device)
	print("model loaded")
	summary_str = str(summary(model, one_mfcc.shape, verbose=0))

	print(summary_str)

	# initialize optimizer and criterion
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	print("criterion: ", criterion)

	n_train_steps = len(train_loader)
	n_val_steps = len(val_loader)

	### loop epochs
	for epoch in range(num_epochs):
		print("\n\ntraining epoch: ", epoch)
		epoch_hist.append(epoch+1)
		epoch_time_start = time.time()
		interval_time_start = time.time()
		model.to(device)

		# at the start of the epoch, set all tracked params to zero
		train_losses = []
		val_losses = []
		inter_epoch_loss = []
		train_num_correct = 0
		val_num_correct = 0

		# set params to be tracked within the epoch ("inter-epoch")
		# these will be outputted at each interval, but not saved
		inter_epoch_num_correct = 0

		### Training loop
		model.train()
		print("model set to train")
		train_preds = []
		train_targets = []
		for i, sample in enumerate(train_loader):

			# prep input and target tensor
			input_tensor = torch.from_numpy(
					np.array(sample['mfccs']).astype(np.float32)).to(device)
			targets = sample['instrument']
			target_tensor = torch.squeeze(torch.tensor(targets), dim=1)
			#print("target tensor after processing: ", target_tensor)
			train_targets.extend(list(targets.numpy()))
			# make predictions
			predictions = torch.squeeze(model(input_tensor).to('cpu'), dim=1)

			# compute loss and do back-propagation
			loss = criterion(predictions, target_tensor)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# append the loss to overall 
			train_losses.append(loss.item())
			inter_epoch_loss.append(loss.item())

			# compute accuracies
			with torch.no_grad():
				predictions_arr = predictions.numpy()
				preds = [np.argmax(predictions_arr[i]) 
					for i in range(len(target_tensor))]
				# inter-epoch accuracy (reset this at each interval)
				inter_epoch_num_correct += np.sum([target_tensor[i] == np.argmax(predictions[i])
					for i in range(len(target_tensor))])
				
				# epoch accuracy (this is tracked and saved)
				train_num_correct += np.sum([target_tensor[i] == np.argmax(predictions[i])
					for i in range(len(target_tensor))])
				#print("debugging in epoch: preds = ", preds)
				train_preds.extend(preds)

			# print step info
			if i % interval == 0:

				# time elapsed
				interval_time_end = time.time()

				# compute mean and std of losses
				inter_epoch_loss_avg = np.mean(inter_epoch_loss)
				inter_epoch_loss_std = np.std(inter_epoch_loss)
				
				# compute inter-epoch accuracy
				# note, this accuracy may be incorrect at the end of each epoch
				# when the batch size is slightly different
				acc = inter_epoch_num_correct / (interval*batch_size)
				print(f"Epoch [{epoch+1}/{num_epochs}], step [{i+1}/{n_train_steps}], ",
							f"Loss: {inter_epoch_loss_avg:.4f} +/- {inter_epoch_loss_std:.4f}, ",
							f"accuracy: {acc}, "
							f"time elapsed = {interval_time_end-interval_time_start}s")
				interval_time_start = time.time()

				# reset inter_epoch metrics
				inter_epoch_num_correct = 0
				inter_epoch_loss = []

		### training loop finished
		# append the accuracy
		train_acc_hist.append(train_num_correct / train_samples)

		# calculate classification metrics
		train_targets = np.array(train_targets).ravel()
		train_preds = np.array(train_preds).ravel()
		# print("debugging: train targets: ", train_targets)
		# print("debugging: train predictions: ", train_preds)
		train_prec, train_recall, train_f1, _ = precision_recall_fscore_support(train_targets, train_preds,
																											average='micro')

		### Validation loop
		model.eval()
		print("model set to eval")
		val_preds = []
		val_targets = []
		with torch.no_grad():

			num_correct = 0
			for i, sample in enumerate(val_loader):
				
				# prep input and target tensor
				input_tensor = torch.from_numpy(
						np.array(sample['mfccs']).astype(np.float32)).to(device)
				targets = sample['instrument']
				val_targets.extend(list(targets.numpy()))
				target_tensor = torch.squeeze(torch.tensor(targets), dim=1)
				#target_tensor = torch.squeeze(torch.tensor(sample['instrument']), dim=1)

				# make predictions
				predictions = torch.squeeze(model(input_tensor).to('cpu'), dim=1)

				# compute and append losses
				loss = criterion(predictions, target_tensor)
				val_losses.append(loss.item())

				predictions_arr = predictions.numpy()
				preds = [np.argmax(predictions_arr[i]) 
					for i in range(len(target_tensor))]
				val_preds.extend(preds)
				# get num correct to comput accuracy
				val_num_correct += np.sum([target_tensor[i] == np.argmax(predictions[i])
					for i in range(len(target_tensor))])
			
			### validation loop finished. prep model and metrics for saving
			# calculate validation accuracy
			val_acc_hist.append(val_num_correct / val_samples)
			val_targets = np.array(val_targets).ravel()
			val_preds = np.array(val_preds).ravel()
			# print("debugging: train targets: ", train_targets)
			# print("debugging: train predictions: ", train_preds)
			val_prec, val_recall, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds,
																											average='micro')
			# calculate mean and standard deviation of losses
			avg_train_loss = np.mean(train_losses)
			std_train_loss = np.std(train_losses)
			avg_val_loss = np.mean(val_losses)
			std_val_loss = np.std(val_losses)

			# append mean and standard deviation to histories
			avg_train_loss_hist.append(avg_train_loss) 
			std_train_loss_hist.append(std_train_loss)  
			avg_val_loss_hist.append(avg_val_loss)
			std_val_loss_hist.append(std_val_loss)

			train_prec_hist.append(train_prec)
			train_recall_hist.append(train_recall)
			train_f1_hist.append(train_f1)

			val_prec_hist.append(val_prec)
			val_recall_hist.append(val_recall)
			val_f1_hist.append(val_f1)
		
		### epoch training finished, output results and save checkpoint

		# text output
		epoch_time_end = time.time()
		print(f"\nEPOCH FINISHED: , ",
					f"training: acc = {train_acc_hist[-1]}, ",
					f"precision = {train_prec_hist[-1]}",
					f"recall = {train_recall_hist[-1]}",
					f"f1 = {train_f1_hist[-1]}",
					f"::: val: acc = {val_acc_hist[-1]}, ",
					f"precision = {val_prec_hist[-1]}",
					f"recall = {val_recall_hist[-1]}",
					f"time elapsed = {epoch_time_end-epoch_time_start}s")
		
		# make a plot
		plt.close("all")
		fig, ax = plt.subplots(ncols=2, figsize=[15, 5])
		#ax.scatter(epoch_hist, avg_train_loss_hist, c='r', label="train loss", )
		ax[0].plot(epoch_hist, avg_train_loss_hist, 'ro--', label="train loss", )
		ax[0].errorbar(x=epoch_hist, y=avg_train_loss_hist, yerr=std_train_loss_hist,
								capsize=5, ls='none', color='r')

		# ax.scatter(epoch_hist, avg_val_loss_hist, c='b', label="val loss", )
		ax[0].plot(epoch_hist, avg_val_loss_hist, 'ko--', label="val loss", )
		ax[0].errorbar(x=epoch_hist, y=avg_val_loss_hist, yerr=std_val_loss_hist,
								capsize=5, ls='none', color='k')
		
		ax[0].set_xlabel("epoch")
		ax[0].set_ylabel("loss")
		ax[0].legend()
		

		ax[1].plot(epoch_hist, train_acc_hist, 'r-.', label="train accuracy", 
									marker='s')

		ax[1].plot(epoch_hist, val_acc_hist, 'k-.', label="val accuracy",
									marker='s')
		ax[1].set_ylabel("accuracy")
		ax[1].set_xlabel("epoch")
		ax[1].set_ylim([0, 1])
		ax[1].legend()
		fig.tight_layout(pad=1)
		plt.show(block=False)
		plt.pause(5)

		# check validation loss if we need to stop training
		# print("validation loss hist: ", avg_val_loss_hist)
		# if (epoch > patience) and all(avg_val_loss_hist[-1-i] >= avg_val_loss_hist[-1-i-1]
		#                               for i in range(patience)):
		model.to('cpu')
		if (epoch > min_epochs) and (
				avg_val_loss_hist[-1] > (std_val_loss_hist[-1] + std_train_loss_hist[-1] + avg_train_loss_hist[-1] + buffer)):
				#avg_val_loss_hist[-1] > (std_val_loss_hist[-1] + std_train_loss_hist[-1] + buffer)):
										# and any(avg_val_loss_hist[-1-i] >= avg_val_loss_hist[-1-i-1]
										#                                   for i in range(patience)):
			notes = notes + "\n\n stopped early"
			# save model
			# TODO: refactor this so torch.save isn't repeated
			if save_checkpoint:
				torch.save({
						'filename': filename,
						'epochs': epoch_hist,
						'model_id': model_id,
						'model_state_dict': model.state_dict(),
						'model_args': model_args,
						'metrics':{
							'avg_train_loss_hist': avg_train_loss_hist,
							'std_train_loss_hist': std_train_loss_hist,
							'avg_val_loss_hist': avg_val_loss_hist,
							'std_val_loss_hist': std_val_loss_hist,
							'train_acc_hist': train_acc_hist,
							'train_prec_hist': train_prec_hist,
							'train_recall_hist': train_recall_hist,
							'train_f1_hist': train_f1_hist,
							'val_acc_hist': val_acc_hist,
							'val_prec_hist': val_prec_hist,
							'val_recall_hist': val_recall_hist,
							'val_f1_hist': val_f1_hist,},
						'dataset_info': dataset.metadata,
						'notes': notes,
						'summary': summary_str,
						'experiment_params': experiment_params,
				}, checkpoint_path+checkpoint_name)
				print("model saved")
				print("stopping early")
				break

		# save model
		if save_checkpoint:
			torch.save({
						'filename': filename,
						'epochs': epoch_hist,
						'model_id': model_id,
						'model_state_dict': model.state_dict(),
						'model_args': model_args,
						'metrics':{
							'avg_train_loss_hist': avg_train_loss_hist,
							'std_train_loss_hist': std_train_loss_hist,
							'avg_val_loss_hist': avg_val_loss_hist,
							'std_val_loss_hist': std_val_loss_hist,
							'train_acc_hist': train_acc_hist,
							'train_prec_hist': train_prec_hist,
							'train_recall_hist': train_recall_hist,
							'train_f1_hist': train_f1_hist,
							'val_acc_hist': val_acc_hist,
							'val_prec_hist': val_prec_hist,
							'val_recall_hist': val_recall_hist,
							'val_f1_hist': val_f1_hist,},
						'dataset_info': dataset.metadata,
						'notes': notes,
						'summary': summary_str,
						'experiment_params': experiment_params,
				}, checkpoint_path+checkpoint_name)
			print("model saved")