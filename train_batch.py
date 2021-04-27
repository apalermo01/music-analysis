import torch
import torch.nn as nn
from train import train_model

"""
call the training loop for a local test run
"""
### constants
ROOT_PATH = "/home/alex/Documents/"
MFCC_FILE = "irmas_data_mfcc13_hop_length256_n_fft2048.json"
CHECKPOINT_PATH = ROOT_PATH + "MIR_trained_models/"

checkpoint_name = "DELETE_THIS_TESTING.pt"

model_args = {
	"channels": [8, 8, 32, 32, 64],
	"conv_kernel_sizes": [3, 3, 3, 3, 3],
	"conv_strides": [1, 1, 1, 1, 1],
	"conv_paddings": [1, 1, 1, 1, 1],
	"pool_masks": [True, True, True, True, True],
	"pool_kernel_sizes": [2, 2, 2, (1, 2), (1, 2)],
	"pool_strides": [2, 2, 2, (1, 2), (1, 2)],
	"linear_features": [128, 64],
	"dropout_probs": [0, 0],
}

args_dict = {
	"filename": MFCC_FILE, 
	"model_id": "Conv_5_layer",
	"num_epochs": 100,
	"interval": 16,
	"lr": 0.001,
	"batch_size": 64,
	"val_split": 0.2,
	"save_checkpoint": True,
	"checkpoint_path": CHECKPOINT_PATH,
	"notes": "Test run",
	"checkpoint_name": checkpoint_name,
	"criterion": nn.CrossEntropyLoss(),
	"patience": 3,
	"min_epochs": 5,
	"buffer": 0.05,
	"model_args": model_args,
}

if __name__ == "__main__":
	train_model(**args_dict)