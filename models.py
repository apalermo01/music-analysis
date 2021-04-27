import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  """Convolutional block with conv2d, linear activation, max pooling, and batch norm
  """
  def __init__(self, in_channels, out_channels, conv_kernel_size=3,
               conv_stride=1, conv_padding=0,
               inc_pool=True, pool_kernel_size=2, pool_stride=2):
    super(ConvBlock, self).__init__()
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

  def forward(self, x):
    x = self.conv_block(x)
    return x

class LinearBlock(nn.Module):
  """Linear block with dense layer, relu, batch norm, then dropout
  """

  def __init__(self, in_features, out_features, dropout_prob=0):
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
  """Linear block with softmax output"""
  def __init__(self, in_features):
    super(HeadBlock, self).__init__()
    self.head_block = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=11),
        nn.Softmax()
    )

  def forward(self, x):
    x = self.head_block(x)
    return x

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