import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
#################################### BLOCKS ####################################
################################################################################

class ConvBlock(nn.Module):

	def __init__(self, in_channels, out_channels, conv_kernel_size=3,
							 conv_stride=1, conv_padding=0,
							 inc_pool=True, pool_kernel_size=2, pool_stride=2):
		"""Convolutional block with conv2d, linear activation, max pooling, 
			and batch norm
		:param in_channels:
		:param out_channels:
		:param conv_kernel_size:
		:param conv_stride:
		:param conv_padding:
		:param inc_pool: If true, includes a max pooling layer

		The following params only matter if inc_pool is True
		:param pool_kernel_size:
		:param pool_stride:
		"""
		super(ConvBlock, self).__init__()

		# construct sequential blocks
		if inc_pool:
			self.conv_block = nn.Sequential(
						nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										kernel_size=conv_kernel_size, stride=conv_stride,
											padding=conv_padding),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
						nn.BatchNorm2d(num_features=out_channels)
				)
		else:
			self.conv_block = nn.Sequential(
						nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
											kernel_size=conv_kernel_size, stride=conv_stride,
											padding=conv_padding),
						nn.ReLU(),
						nn.BatchNorm2d(num_features=out_channels)
				) 

	# run forward
	def forward(self, x):
		x = self.conv_block(x)
		return x

class LinearBlock(nn.Module):

	def __init__(self, in_features, out_features, dropout_prob=0):
		"""Linear block with dense layer, relu, batch norm, then dropout
		:param in_features:
		:param out_features:
		:param dropout_prob: Set to 0 for no dropout layer
		"""

		super(LinearBlock, self).__init__()
		self.linear_block = nn.Sequential(
				nn.Linear(in_features=in_features, out_features=out_features),
				nn.ReLU(),
				nn.BatchNorm1d(num_features=out_features),
				nn.Dropout(p=dropout_prob)
		)

	def forward(self, x):
		x = self.linear_block(x)
		return x

class HeadBlock(nn.Module):

	def __init__(self, in_features):
		"""Linear block with softmax output.
		NOTE: no longer using softmax output since the CrossEntropyLoss handles that
		:param in_features:
		out_features is fixed to 11 to corrospond to the number of classes
		"""
		super(HeadBlock, self).__init__()
		self.head_block = nn.Sequential(
				nn.Linear(in_features=in_features, out_features=11),
				#nn.Softmax()
		)

	def forward(self, x):
		x = self.head_block(x)
		return x

################################################################################
################################### NETWORKS ###################################
################################################################################
class Conv1Layer(nn.Module):

	def __init__(self, single_sample, channels=[8],
							 conv_kernel_sizes=[3],
							 conv_strides=[1],
							 conv_paddings=[0],
							 pool_masks=[True],
							 pool_kernel_sizes=[2],
							 pool_strides=[2],
							 linear_features=[128, 64],
							 dropout_probs=[0, 0]):

		"""Convolutional neural network with 1 conv layer and 3 linear layers.
		All hyperparams are flexible and initialized using lists (or array-likes).
		The nth entry in each list corrosponds to the nth layer

		:param single_sample: a sample mfcc to run through the network on init to 
		get layer sizes
		:param channels:
		:param conv_kernel_sizes:
		:param conv_paddings:
		:param pool_masks: array of booleans to control max pooling
			ex: [False, True] means no max pooling after 1st layer, but max pooling 
			after second layer. Other hyperparams for maxpooling must be passed so
			that alignment is consistent. ex: in the [False, True] example, one could
			pass [3, 2] for pool kernel size. The 3 does nothing but the 2 will use 
			a pool kernel size of 2. Passing only [2] will result in an error even if 
			there is only one maxpool layer.
		:param pool_kernel_sizes:
		:param pool_strides:
		:param linear features: output sizes for linear layers (input size
			determined on init by one_mfcc)
		:param dropout_probs:

		"""
		super(Conv1Layer, self).__init__()

		# convolutional blocks
		self.conv1 = ConvBlock(in_channels=1, out_channels=channels[0],
													 conv_kernel_size=conv_kernel_sizes[0],
													 conv_stride=conv_strides[0],
													 conv_padding=conv_paddings[0],
													 inc_pool=pool_masks[0],
													 pool_kernel_size=pool_kernel_sizes[0],
													 pool_stride=pool_strides[0])
	
		# run a single sample through the convolutional block to get output size
		# https://discuss.pytorch.org/t/convolution-and-pooling-layers-need-a-method-to-calculate-output-size/21895
		sample_output1 = self.conv1(torch.from_numpy(
				single_sample[np.newaxis,...].astype(np.float32)))
	
		sample_flattened = sample_output1.flatten(start_dim=1)
 
		# linear blocks
		self.linear1 = LinearBlock(in_features=(sample_flattened.shape[1]),
																						out_features=(linear_features[0]),
																						dropout_prob=dropout_probs[0])
		self.linear2 = LinearBlock(in_features=(linear_features[0]),
																						out_features=(linear_features[1]),
																						dropout_prob=dropout_probs[1])
		self.head = HeadBlock(in_features=(linear_features[1]))

	def forward(self, x):
		x = self.conv1(x)
		x = x.flatten(start_dim=1)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.head(x)
		return x

class Conv3Layer(nn.Module):
	def __init__(self, single_sample, channels=[8, 16, 32],
							 conv_kernel_sizes=[3, 3, 3],
							 conv_strides=[1, 1, 1],
							 conv_paddings=[0, 0, 1,],
							 pool_masks=[True, False, False],
							 pool_kernel_sizes=[2, 2, 2],
							 pool_strides=[2, 2, 2],
							 linear_features=[128, 64],
							 dropout_probs=[0, 0]):

		"""Convolutional neural network with 3 conv layers and 3 linear layers.
		All hyperparams are flexible and initialized using lists (or array-likes).
		The nth entry in each list corrosponds to the nth layer

		:param single_sample: a sample mfcc to run through the network on init to 
		get layer sizes
		:param channels:
		:param conv_kernel_sizes:
		:param conv_paddings:
		:param pool_masks: array of booleans to control max pooling
			ex: [False, True] means no max pooling after 1st layer, but max pooling 
			after second layer. Other hyperparams for maxpooling must be passed so
			that alignment is consistent. ex: in the [False, True] example, one could
			pass [3, 2] for pool kernel size. The 3 does nothing but the 2 will use 
			a pool kernel size of 2. Passing only [2] will result in an error even if 
			there is only one maxpool layer.
		:param pool_kernel_sizes:
		:param pool_strides:
		:param linear features: output sizes for linear layers (input size
			determined on init by one_mfcc)
		:param dropout_probs:
		"""

		super(Conv3Layer, self).__init__()

		self.conv1 = ConvBlock(in_channels=1, out_channels=channels[0],
													 conv_kernel_size=conv_kernel_sizes[0],
													 conv_stride=conv_strides[0],
													 conv_padding=conv_paddings[0],
													 inc_pool=pool_masks[0],
													 pool_kernel_size=pool_kernel_sizes[0],
													 pool_stride=pool_strides[0])
		
		self.conv2 = ConvBlock(in_channels=channels[0], out_channels=channels[1],
													 conv_kernel_size=conv_kernel_sizes[1],
													 conv_stride=conv_strides[1],
													 conv_padding=conv_paddings[1],
													 inc_pool=pool_masks[1],
													 pool_kernel_size=pool_kernel_sizes[1],
													 pool_stride=pool_strides[1])

		self.conv3 = ConvBlock(in_channels=channels[1], out_channels=channels[2],
													 conv_kernel_size=conv_kernel_sizes[2],
													 conv_stride=conv_strides[2],
													 conv_padding=conv_paddings[2],
													 inc_pool=pool_masks[2],
													 pool_kernel_size=pool_kernel_sizes[2],
													 pool_stride=pool_strides[2])
		
		# calculate size for linear layers
		sample_output1 = self.conv1(torch.from_numpy(
				single_sample[np.newaxis,...].astype(np.float32)))
		sample_output2 = self.conv2(sample_output1)
		sample_output3 = self.conv3(sample_output2)
		sample_flattened = sample_output3.flatten(start_dim=1)

		# linear blocks
		self.linear1 = LinearBlock(in_features=(sample_flattened.shape[1]),
																						out_features=(linear_features[0]),
																						dropout_prob=dropout_probs[0])
		self.linear2 = LinearBlock(in_features=(linear_features[0]),
																						out_features=(linear_features[1]),
																						dropout_prob=dropout_probs[1])
		self.head = HeadBlock(in_features=(linear_features[1]))
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.flatten(start_dim=1)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.head(x)
		return x

class Conv5Layer(nn.Module):
	def __init__(self, single_sample, channels= [8, 8, 32, 32, 64],
							 conv_kernel_sizes=[3, 3, 3, 3, 3],
							 conv_strides=[1, 1, 1, 1, 1],
							 conv_paddings=[0, 0, 1, 1, 1],
							 pool_masks=[True, False, False, False, False],
							 pool_kernel_sizes=[2, 2, 2, 2, 2],
							 pool_strides=[2, 2, 2, 2, 2],
							 linear_features=[128, 64],
							 dropout_probs=[0, 0]):
		
		"""Convolutional neural network with 3 conv layers and 3 linear layers.
		All hyperparams are flexible and initialized using lists (or array-likes).
		The nth entry in each list corrosponds to the nth layer

		:param single_sample: a sample mfcc to run through the network on init to 
		get layer sizes
		:param channels:
		:param conv_kernel_sizes:
		:param conv_paddings:
		:param pool_masks: array of booleans to control max pooling
			ex: [False, True] means no max pooling after 1st layer, but max pooling 
			after second layer. Other hyperparams for maxpooling must be passed so
			that alignment is consistent. ex: in the [False, True] example, one could
			pass [3, 2] for pool kernel size. The 3 does nothing but the 2 will use 
			a pool kernel size of 2. Passing only [2] will result in an error even if 
			there is only one maxpool layer.
		:param pool_kernel_sizes:
		:param pool_strides:
		:param linear features: output sizes for linear layers (input size
			determined on init by one_mfcc)
		:param dropout_probs:
		"""

		super(Conv5Layer, self).__init__()

		# convolutional layers
		self.conv1 = ConvBlock(in_channels=1, out_channels=channels[0],
													 conv_kernel_size=conv_kernel_sizes[0],
													 conv_stride=conv_strides[0],
													 conv_padding=conv_paddings[0],
													 inc_pool=pool_masks[0],
													 pool_kernel_size=pool_kernel_sizes[0],
													 pool_stride=pool_strides[0])
		
		self.conv2 = ConvBlock(in_channels=channels[0], out_channels=channels[1],
													 conv_kernel_size=conv_kernel_sizes[1],
													 conv_stride=conv_strides[1],
													 conv_padding=conv_paddings[1],
													 inc_pool=pool_masks[1],
													 pool_kernel_size=pool_kernel_sizes[1],
													 pool_stride=pool_strides[1])

		self.conv3 = ConvBlock(in_channels=channels[1], out_channels=channels[2],
													 conv_kernel_size=conv_kernel_sizes[2],
													 conv_stride=conv_strides[2],
													 conv_padding=conv_paddings[2],
													 inc_pool=pool_masks[2],
													 pool_kernel_size=pool_kernel_sizes[2],
													 pool_stride=pool_strides[2])
		
		self.conv4 = ConvBlock(in_channels=channels[2], out_channels=channels[3],
													 conv_kernel_size=conv_kernel_sizes[3],
													 conv_stride=conv_strides[3],
													 conv_padding=conv_paddings[3],
													 inc_pool=pool_masks[3],
													 pool_kernel_size=pool_kernel_sizes[3],
													 pool_stride=pool_strides[3])
		
		self.conv5 = ConvBlock(in_channels=channels[3], out_channels=channels[4],
													 conv_kernel_size=conv_kernel_sizes[4],
													 conv_stride=conv_strides[4],
													 conv_padding=conv_paddings[4],
													 inc_pool=pool_masks[4],
													 pool_kernel_size=pool_kernel_sizes[4],
													 pool_stride=pool_strides[4])
		
		# calculate size for linear layers
		sample_output1 = self.conv1(torch.from_numpy(
				single_sample[np.newaxis,...].astype(np.float32)))
		sample_output2 = self.conv2(sample_output1)
		sample_output3 = self.conv3(sample_output2)
		sample_output4 = self.conv4(sample_output3)
		sample_output5 = self.conv5(sample_output4)
		sample_flattened = sample_output5.flatten(start_dim=1)


		# linear blocks
		self.linear1 = LinearBlock(in_features=(sample_flattened.shape[1]),
																						out_features=(linear_features[0]),
																						dropout_prob=dropout_probs[0])
		self.linear2 = LinearBlock(in_features=(linear_features[0]),
																						out_features=(linear_features[1]),
																						dropout_prob=dropout_probs[1])
		self.head = HeadBlock(in_features=(linear_features[1]))
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = x.flatten(start_dim=1)
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.head(x)
		return x

class ConvNLayer(nn.Module):
	def __init__(self, single_sample, 

		num_conv_layers=2,
		channels=[8, 16],
		conv_kernel_sizes=[3, 3],
		conv_strides=[1, 1],
		conv_paddings=[1, 1],
		pool_masks=[True, True],
		pool_kernel_sizes=[2, 2],
		pool_strides=[2, 2],
		
		num_linear_layers=2,
		linear_features=[128, 64],
		dropout_probs=[0, 0]
		):
		"""Convolutional neural net with an arbitrary number of convolutional layers
		"""
		super(ConvNLayer, self).__init__()

		self.num_conv_layers = num_conv_layers
		self.num_linear_layers = num_linear_layers

		# prepend 1 to input channels since there is only one
		channels.insert(0, 1)

		# define list of convolutional layers
		self.conv_layers = [
			ConvBlock(
				in_channels = channels[i],
				out_channels = channels[i+1],
				conv_kernel_size = conv_kernel_sizes[i],
				conv_stride = conv_strides[i],
				conv_padding = conv_paddings[i],
				inc_pool = pool_masks[i],
				pool_kernel_size = pool_kernel_sizes[i],
				pool_stride = pool_strides[i])
		for i in range(self.num_conv_layers)]

		# calculate size of linear layers
		sample = torch.from_numpy(
			single_sample[np.newaxis,...].astype(np.float32)
		)

		for i in range(self.num_conv_layers):
			sample = self.conv_layers[i](sample)

		sample_flattened = sample.flatten(start_dim=1)

		# prepend shape of input to linear block
		linear_features.insert(0, sample_flattened.shape[1])

		# define list of linear layers
		self.linear_layers = [
			LinearBlock(
				in_features = (linear_features[i]),
				out_features = (linear_features[i+1]),
				dropout_prob = dropout_probs[i])
			for i in range(self.num_linear_layers)
		]

		# define output head
		self.head = HeadBlock(in_features=(linear_features[-1]))

	def forward(self, x):
		x.to('cpu')
		for i in range(self.num_conv_layers):
			print(self.conv_layers[i])
			x = self.conv_layers[i](x)
		
		x = x.flatten(start_dim=1)

		for i in range(self.num_linear_layers):
			x = self.linear_layers[i](x)
		
		x = self.head(x)
		return x