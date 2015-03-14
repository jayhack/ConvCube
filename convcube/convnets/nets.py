import numpy as np
import scipy as sp

from convcube.cs231n.layers import *
from convcube.cs231n.fast_layers import *
from convcube.cs231n.layer_utils import *

def euclidean_loss(y_pred, y):
	"""returns summed loss, gradients for each sample"""
	assert y_pred.shape == y.shape

	#=====[ Step 1: Loss ]=====
	diffs = y_pred - y
	loss = np.mean(np.sqrt(np.sum(diffs**2, axis=1))) / 2

	#=====[ Step 2: Gradients ]=====
	grads = diffs
	return loss, grads


def init_six_layer_net(pretrained_model=None,
							input_shape=(3, 64, 64),
							num_filters=(32, 16, 16, 16, 32),
							filter_sizes=(3, 3, 3, 3),
							size_out=2,
							weight_scale=1e-2,
							bias_scale=0,
							dtype=np.float32
						):
	"""
	Initialize a three layer ConvNet with the following architecture:

	conv - relu - pool - affine - relu - dropout - affine - euclidean_loss

	(outputs coordinates in R2 corresponding to Cube location)

	The convolutional layer uses stride 1 and has padding to perform "same"
	convolution, and the pooling layer is 2x2 stride 2.

	Inputs:
	- input_shape: Tuple (C, H, W) giving the shape of each training sample.
	  Default is (3, 32, 32) for CIFAR-10.
	- filter_size: The height and width of filters in the convolutional layer.
	- num_filters: Tuple (F, H) where F is the number of filters to use in the
	  convolutional layer and H is the number of neurons to use in the hidden
	  affine layer.
	- weight_scale: Weights are initialized from a gaussian distribution with
	  standard deviation equal to weight_scale.
	- bias_scale: Biases are initialized from a gaussian distribution with
	  standard deviation equal to bias_scale.
	- dtype: Numpy datatype used to store parameters. Default is float32 for
	  speed.
	"""
	C, H, W = input_shape
	F1, F2, F3, F4, FC = num_filters
	filter_sizes = filter_sizes
	model = {}

	#=====[ Step 1: Set weights from pretrained/std normal 	]=====
	if pretrained_model:
		model['W1'] = pretrained_model['W1']	#copy pretrained model bottom conv layer
		model['b1'] = pretrained_model['b1']

		model['W2'] = pretrained_model['W2']
		model['b2'] = pretrained_model['b2']
	else:
		model['W1'] = np.random.randn(F1, 3, filter_sizes[0], filter_sizes[0]) # Will reuse
		model['b1'] = np.random.randn(F1)

		#####[ CONV-RELU-POOL	]#####
		model['W2'] = np.random.randn(F2, F1, filter_sizes[1], filter_sizes[1]) # Will reuse
		model['b2'] = np.random.randn(F2)

	model['W3'] = np.random.randn(F3, F2, filter_sizes[2], filter_sizes[2])
	model['b3'] = np.random.randn(F3)

	model['W4'] = np.random.randn(F4, F3, filter_sizes[3], filter_sizes[3])
	model['b4'] = np.random.randn(F4)

	model['W5'] = np.random.randn(F4*(H/(2*2*2*2))*(W/(2*2*2*2)), FC)
	model['b5'] = np.random.randn(FC)

	model['W6'] = np.random.randn(FC, size_out)
	model['b6'] = np.random.randn(size_out)

	#=====[ Step 2: Scale weights	]=====
	regs = [2, 3, 4, 5, 6]
	if pretrained_model:
		regs.insert(0, 1)
	for i in regs:
		model['W%d' % i] *= weight_scale
		model['b%d' % i] *= bias_scale

	#=====[ Step 3: Convert to dtype	]=====
	for k in model:
		model[k] = model[k].astype(dtype, copy=False)

	return model


def six_layer_net(X, model, y=None, reg=0.0, dropout=None):
	"""
	Compute the loss and gradient for a simple three layer ConvNet that uses
	the following architecture:

	conv - relu - pool - affine - relu - dropout - affine - euclidean loss

	(loss is from actual coordinates)

	The convolution layer uses stride 1 and sets the padding to achieve "same"
	convolutions, and the pooling layer is 2x2 stride 2. We use L2 regularization
	on all weights, and no regularization on the biases.

	Inputs:
	- X: (N, C, H, W) array of input data
	- model: Dictionary mapping parameter names to values; it should contain
	the following parameters:
	- W1, b1: Weights and biases for convolutional layer
	- W2, b2, W3, b3: Weights and biases for affine layers
	- y: Integer array of shape (N,) giving the labels for the training samples
	in X. This is optional; if it is not given then return classification
	scores; if it is given then instead return loss and gradients.
	- reg: The regularization strength.
	- dropout: The dropout parameter. If this is None then we skip the dropout
	layer; this allows this function to work even before the dropout layer
	has been implemented.
	"""
	#=====[ Step 1: Initialization	]=====
	W1, b1 = model['W1'], model['b1']
	W2, b2 = model['W2'], model['b2']
	W3, b3 = model['W3'], model['b3']
	W4, b4 = model['W4'], model['b4']
	W5, b5 = model['W5'], model['b5']	
	W6, b6 = model['W6'], model['b6']
	conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
	conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
	conv_param_3 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}
	conv_param_4 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}
	pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
	dropout_param = {'p': dropout}
	dropout_param['mode'] = 'test' if y is None else 'train'

	#=====[ Step 2: Forward pass	]=====
	a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
	a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
	a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param_3, pool_param)
	a4, cache4 = conv_relu_pool_forward(a3, W4, b4, conv_param_4, pool_param)
	a5, cache5 = affine_relu_forward(a4, W5, b5)
	scores, cache6 = affine_forward(a5, W6, b6)

	#=====[ Step 3: Return scores during feedforward inference	]=====
	if y is None:
		return scores

	#=====[ Step 4: Backward pass - loss and gradients	]=====
	data_loss, dscores = euclidean_loss(scores, y)
	da5, dW6, db6 = affine_backward(dscores, cache6)
	da4, dW5, db5 = affine_relu_backward(da5, cache5)
	da3, dW4, db4 = conv_relu_pool_backward(da4, cache4)
	da2, dW3, db3 = conv_relu_pool_backward(da3, cache3)
	da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
	dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)



	#=====[ Step 5: regularize gradients, get total loss	]=====
	grads = { 
				'W1': dW1, 'b1': db1, 
				'W2': dW2, 'b2': db2, 
				'W3': dW3, 'b3': db3,
				'W4': dW4, 'b4': db4,
				'W5': dW5, 'b5': db5,
				'W6': dW6, 'b6': db6
			}
	reg_loss = 0.0
	for p in ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']:
		W = model[p]
		reg_loss += 0.5 * reg * np.sum(W * W)
		grads[p] += reg * W
	loss = data_loss + reg_loss

	return loss, grads