import torch
import torch.nn as nn
from train import train_model
from textwrap import dedent

"""
call the training loop for a local test run
"""
### constants
ROOT_PATH = "/home/alex/Documents/"
MFCC_FILE = "irmas_data_mfcc13_hop_length256_n_fft2048.json"
CHECKPOINT_PATH = ROOT_PATH + "MIR_trained_models/"



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
	"model_id": "Conv_N_layer",
	"num_epochs": 100,
	"interval": 16,
	"lr": 0.01,
	"batch_size": 32,
	"val_split": 0.2,
	"save_checkpoint": False,
	"checkpoint_path": CHECKPOINT_PATH,
	"notes": "Test run",
	"checkpoint_name": "DELETE_THIS.pt",
	"criterion": nn.CrossEntropyLoss(),
	"patience": 2,
	"min_epochs": 3,
	"buffer": 0.01,
	"model_args": model_args,
}

if __name__ == "__main__":
	model_ids = ["Conv_1_layer", "Conv_2_layer", "Conv_3_layer"]
	pool_types = ["sym_stride", "asym_stride"]
	pool_strides = [
		[
			[(2, 2)], [(1, 2)]
		],
		[
			[(2, 2), (2, 2), (2, 2)], [(1, 2), (1, 2), (1, 2)]
		],
		[
			[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
			[(1, 2), (1, 2), (1, 2), (1, 2), (1, 2)]
		]
	]
	channels = [
		[
			[8], [16], [32], [64], [128]
		],
		[
			[8, 16, 32], [8, 16, 64], [8, 32, 64], [8, 32, 128],
			[16, 32, 64], [16, 64, 128], [128, 128, 128]
		],
		[
			[8, 16, 32, 64, 128], [8, 16, 16, 32, 64], [8, 32, 64, 64, 128,],
			[16, 16, 32, 64, 128], [16, 32, 32, 64, 128], [32, 32, 64, 64, 128]
		]
	]

	for i, id in enumerate(model_ids):
		for k, channel in enumerate(channels[i]):
			for j, stride in enumerate(pool_types):
				print("stride: ", pool_strides[i][j])
				model_args = {
				"channels": channel,
				"conv_kernel_sizes": [3, 3, 3, 3, 3],
				"conv_strides": [1, 1, 1, 1, 1],
				"conv_paddings": [1, 1, 1, 1, 1],
				"pool_masks": [True, True, True, True, True],
				"pool_kernel_sizes": pool_strides[i][j],
				"pool_strides": pool_strides[i][j],
				"linear_features": [128, 64],
				"dropout_probs": [0.3, 0.3],
				}
				args_dict['model_id'] = IsADirectoryError
				args_dict['model_args'] = model_args
				args_dict['experiment_params'] = {
					'channels': channel,
					'pool_kernel_sizes': pool_strides[i][j],
					'stride_type': stride,
				}
				args_dict['notes'] = dedent("""
				varying stride and channel depth with dropout prob=0.3.

				Other hyperparams:
				lr = 0.01
				interval: 16
				batch_size: 32
				criterion: CrossEntropyLoss
				
				""")
				args_dict['checkpoint_name'] = "{}_channel{}_{}.pt".format(
					id, k,stride,)
				train_model(**args_dict)