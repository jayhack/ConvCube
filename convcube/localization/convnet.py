import numpy as np
import scipy as sp
from convcube.cs231n.classifiers.convnet import init_three_layer_convnet
from convcube.cs231n.classifiers.convnet import three_layer_convnet
from convcube.cs231n.classifier_trainer import ClassifierTrainer
from convcube.cs231n.layers import *
from convcube.cs231n.fast_layers import *
from convcube.cs231n.layer_utils import *
from training import LocalizationConvNetTrainer
from utils import *


class LocalizationConvNet(object):
	"""
	Class: LocalizationConvNet
	--------------------------
	ConvNet for finding the locale of the Cube 

	Args:
	-----
	- shape_pre: shape of pretraining data. 
	  Default: (3, 64, 64) (ImageNet size)
	- shape_loc: shape of data to perform transfer learning and localization 
	  inference on.
	  Default: (3, 46, 80)
	- classes_pre: # of classes in pretraining dataset
	- filter_size: height/width of filters in conv layer
	  Default: 5
	- num_filters_pretrain: Tuple (F, H) where F is the number of filters to 
	  use in the convolutional layer and H is the number of neurons to use 
	  in the hidden affine layer during pretraining.
	- num_filters_transfer: Same as pretrain; just change H if curious
      Default: (10,10)
	  on. default 
	"""

	def __init__(self, 	
						shape_pre=(3, 64, 64),
						shape_loc=(3, 46, 80),
						classes_pre=100,
						filter_size=5,
						num_filters_pre=(10,10),
						num_filters_loc=(10,10),
						weight_scale=1e-2,
						bias_scale=0,
						dtype=np.float32
				):
		self.shape_pre = shape_pre
		self.shape_loc = shape_loc
		self.classes_pre = classes_pre
		self.filter_size = filter_size
		self.num_filters_pre = num_filters_pre
		self.num_filters_loc = num_filters_loc
		assert num_filters_pre[0] == num_filters_loc[0]
		self.weight_scale = weight_scale
		self.bias_scale = bias_scale
		self.dtype = dtype


	################################################################################
	####################[ Pretraining ]#############################################
	################################################################################

	def init_pretrain_convnet(self):
		"""
		    Initialize a three layer ConvNet with the following architecture:

		    conv - relu - pool - affine - relu - dropout - affine - softmax

		    The convolutional layer uses stride 1 and has padding to perform "same"
		    convolution, and the pooling layer is 2x2 stride 2.
    	"""

		model = init_three_layer_convnet(	
											input_shape=self.shape_pre,
											num_classes=self.classes_pre,
											filter_size=self.filter_size,
											num_filters=self.num_filters_pre,
											weight_scale=self.weight_scale,
											bias_scale=self.bias_scale,
											dtype=self.dtype
										)
		return model


	def pretrain(self, X_train, y_train, X_val, y_val, verbose=True):
		"""
			trains pretrain convnet. 

			returns model, loss_hist, train_acc_hist, val_acc_hist
		"""
		model = self.init_pretrain_convnet()
		trainer = ClassifierTrainer()
		result = trainer.train(
		 						X_train, y_train, 
		 						X_val, y_val, 
		 						model, 
	 							three_layer_convnet, 
		 						dropout=None, reg=0.05, learning_rate=0.00005, 
		 						batch_size=50, num_epochs=8,
								learning_rate_decay=1.0, update='rmsprop', verbose=verbose
								)
		model, loss_hist, train_acc_hist, val_acc_hist = result
		return model, loss_hist, train_acc_hist, val_acc_hist



	################################################################################
	####################[ Localization ]############################################
	################################################################################

	def init_localization_convnet(self, pretrained_model):
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
		C, H, W = self.shape_loc
		F1, FC = self.num_filters_loc
		model = {}

		#=====[ Step 1: Set weights from pretrained/std normal 	]=====
		model['W1'] = pretrained_model['W1']
		model['b1'] = pretrained_model['b1']
		model['W2'] = np.random.randn(H * W * F1 / 4, FC)
		model['b2'] = np.random.randn(FC)
		model['W3'] = np.random.randn(FC, 2)
		model['b3'] = np.random.randn(2)

		#=====[ Step 2: Scale weights	]=====
		for i in [2, 3]:
			model['W%d' % i] *= self.weight_scale
			model['b%d' % i] *= self.bias_scale

		#=====[ Step 3: Convert to dtype	]=====
		for k in model:
			model[k] = model[k].astype(self.dtype, copy=False)

		return model


	def euclidean_loss(self, y_pred, y):
		"""returns summed loss, gradients for each sample"""
		assert y_pred.shape == y.shape

		#=====[ Step 1: Loss ]=====
		diffs = y_pred - y
		loss = np.sum(np.sum(diffs**2, axis=1))

		#=====[ Step 2: Gradients ]=====
		grads = 2 * diffs
		return loss, grads


	def localization_convnet(self, X, model, y=None, reg=0.0, dropout=None):
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
		conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
		pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
		dropout_param = {'p': dropout}
		dropout_param['mode'] = 'test' if y is None else 'train'

		#=====[ Step 2: Forward pass	]=====
		a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
		a2, cache2 = affine_relu_forward(a1, W2, b2)
		if dropout is None:
			scores, cache4 = affine_forward(a2, W3, b3)
		else:
			d2, cache3 = dropout_forward(a2, dropout_param)
			scores, cache4 = affine_forward(d2, W3, b3)

		#=====[ Step 3: Return scores during feedforward inference	]=====
		if y is None:
			return scores

		#=====[ Step 4: Backward pass - loss and gradients	]=====
		data_loss, dscores = self.euclidean_loss(scores, y)
		if dropout is None:
			da2, dW3, db3 = affine_backward(dscores, cache4)
		else:
			dd2, dW3, db3 = affine_backward(dscores, cache4)
			da2 = dropout_backward(dd2, cache3)
		da1, dW2, db2 = affine_relu_backward(da2, cache2)
		dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

		#=====[ Step 5: regularize gradients, get total loss	]=====
		grads = { 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3 }
		reg_loss = 0.0
		for p in ['W1', 'W2', 'W3']:
			W = model[p]
			reg_loss += 0.5 * reg * np.sum(W * W)
			grads[p] += reg * W
		loss = data_loss + reg_loss

		return loss, grads



	def train_localization_convnet(self, X_train, y_train, X_val, y_val, pretrained_model):
		"""
			trains localization convnet. 

			returns model, loss_hist, train_acc_hist, val_acc_hist
		"""
		trainer = LocalizationConvNetTrainer()
		loc_model = self.init_localization_convnet(pretrained_model)
		model, loss_hist, train_acc_hist, val_acc_hist = trainer.train(
																		X_train, y_train, 
																		X_val, y_val, 
																		loc_model, self.localization_convnet, 
																		dropout=None, reg=0.05, 
																		learning_rate=0.00005, 
																		batch_size=50, num_epochs=100,
																		learning_rate_decay=1.0, 
																		update='rmsprop', verbose=True
																	)
		return model, loss_hist, train_acc_hist, val_acc_hist
















