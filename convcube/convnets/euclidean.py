import pickle
import numpy as np
import scipy as sp
from convcube.cs231n.classifiers.convnet import init_three_layer_convnet
from convcube.cs231n.classifiers.convnet import three_layer_convnet
from convcube.cs231n.classifiers.convnet import init_five_layer_convnet
from convcube.cs231n.classifiers.convnet import five_layer_convnet
from convcube.cs231n.classifier_trainer import ClassifierTrainer as PretrainClassifierTrainer

from convcube.cs231n.layers import *
from convcube.cs231n.fast_layers import *
from convcube.cs231n.layer_utils import *
from euclidean_trainer import EuclideanConvNetTrainer


class EuclideanConvNet(object):
	"""
	Class: HeatmapConvNet
	---------------------
	ConvNet for finding a heatmap of corners from the cube

	Args:
	-----
	- shape_pre: shape of pretraining data. 
	  Default: (3, 64, 64) (ImageNet size)
	- shape_loc: shape of data to perform transfer learning and localization 
	  inference on.
	  Default: (3, 46, 80)
	- classes_pre: # of classes in pretraining dataset
	- size_out: # of points to project onto
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
						pretrained_model=None,
						model=None,
						shape_pre=(3, 64, 64),
						shape_loc=(3, 46, 80),
						classes_pre=100,
						size_out=4,
						filter_sizes_pre=(10, 5, 5),
						filter_sizes_loc=(8, 5, 5),
						num_filters_pre=(10,10,32),
						num_filters_loc=(10,10,32),
						weight_scale=1e-2,
						bias_scale=0,
						dtype=np.float32
				):

		self.pretrained_model = pretrained_model
		self.model = model

		assert num_filters_pre[0] == num_filters_loc[0]
		self.params = {
						'shape_pre':shape_pre,
						'shape_loc':shape_loc,
						'classes_pre':classes_pre,
						'size_out':size_out,
						'filter_sizes_pre':filter_sizes_pre,
						'filter_sizes_loc':filter_sizes_loc,
						'num_filters_pre':num_filters_pre,
						'num_filters_loc':num_filters_loc,
						'weight_scale':weight_scale,
						'bias_scale':bias_scale,
						'dtype':dtype,
					}


	################################################################################
	####################[ Pretraining ]#############################################
	################################################################################

	def init_pretrain_convnet(self):
		"""
		    Initialize a three layer ConvNet with the following architecture:

		    five-layer convnet, with two conv-relu-pool, then affine-relu, then affine

		    The convolutional layer uses stride 1 and has padding to perform "same"
		    convolution, and the pooling layer is 2x2 stride 2.

		    Only the first three layers will get reused
		"""
		C, H, W = self.params['shape_pre']
		F1, F2, FC = self.params['num_filters_pre']
		filter_sizes = self.params['filter_sizes_pre']
		num_classes = self.params['classes_pre']
		model = {}

		#=====[ Step 1: Set weights from pretrained/std normal 	]=====
		#####[ CONV-RELU-POOL	]#####
		model['W1'] = np.random.randn(F1, 3, filter_sizes[0], filter_sizes[0]) # Will reuse
		model['b1'] = np.random.randn(F1)

		#####[ CONV-RELU-POOL	]#####
		#64 -> 63
		model['W2'] = np.random.randn(F2, F1, filter_sizes[1], filter_sizes[1]) # Will reuse
		model['b2'] = np.random.randn(F2)

		#63 -> 62? 64?
		######[ AFFINE-RELU FULLY CONNECTED	]#####
		#TODO: what should that parameter be in the fully connected layer?
		#goes from output of W2 to FC, so we need to find the shape of W2
		model['W3'] = np.random.randn(F2*(64/(2*2))*(64/(2*2)), FC) #down to 16 from 64 due to 2x 2x2 pooling.
		model['b3'] = np.random.randn(FC)

		#####[ CONV-RELU-POOL	]#####
		# model['W3'] = np.random.randn(H * W * F2 / 4, FC) #fully connected layer
		model['W4'] = np.random.randn(FC, num_classes)
		model['b4'] = np.random.randn(num_classes)


		#=====[ Step 2: Scale weights	]=====
		for i in [1, 2, 3, 4]:
			model['W%d' % i] *= self.params['weight_scale']
			model['b%d' % i] *= self.params['bias_scale']

		#=====[ Step 3: Convert to dtype	]=====
		for k in model:
			model[k] = model[k].astype(self.params['dtype'], copy=False)

		return model


	def pretrain_convnet(self, X, model, y=None, reg=0.0, dropout=None):
		"""runs the pretrain convnet"""
		#=====[ Step 1: initialization	]=====
		W1, b1 = model['W1'], model['b1']
		W2, b2 = model['W2'], model['b2']
		W3, b3 = model['W3'], model['b3']
		W4, b4 = model['W4'], model['b4']
		conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
		conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
		pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
		dropout_param = {'p': dropout}
		dropout_param['mode'] = 'test' if y is None else 'train'

		#=====[ Step 2: forward pass	]=====
		a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
		a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
		a3, cache3 = affine_relu_forward(a2, W3, b3)
		scores, cache4 = affine_forward(a3, W4, b4)

		#=====[ Step 3: scores if not training	]=====
		if y is None:
			return scores

		#=====[ Step 4: backward pass	]=====
		data_loss, dscores = softmax_loss(scores, y)
		da3, dW4, db4 = affine_backward(dscores, cache4)
		da2, dW3, db3 = affine_relu_backward(da3, cache3)
		da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
		dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)
	
		grads = { 
					'W1': dW1, 'b1': db1, 
					'W2': dW2, 'b2': db2, 
					'W3': dW3, 'b3': db3,
					'W4': dW4, 'b4': db4 
				}

		reg_loss = 0.0
		for p in ['W1', 'W2', 'W3']:
			W = model[p]
			reg_loss += 0.5 * reg * np.sum(W * W)
			grads[p] += reg * W
		loss = data_loss + reg_loss
		return loss, grads


	def pretrain(self, 
						X_train, y_train, X_val, y_val, verbose=True,
						dropout=1.0, reg=0.05, learning_rate=0.0005, 
		 				batch_size=50, num_epochs=3,
						learning_rate_decay=0.95, update='rmsprop'
					):
		"""
			trains pretrain convnet. 

			returns model, loss_hist, train_acc_hist, val_acc_hist
		"""
		model = self.init_pretrain_convnet()
		trainer = PretrainClassifierTrainer()
		result = trainer.train(
		 						X_train, y_train, 
		 						X_val, y_val, 
		 						model, 
	 							self.pretrain_convnet,
		 						dropout=dropout, reg=reg, learning_rate=learning_rate, 
		 						batch_size=batch_size, num_epochs=num_epochs,
								learning_rate_decay=learning_rate_decay, 
								update=update, verbose=verbose
								)
		pretrained_model, loss_hist, train_acc_hist, val_acc_hist = result
		self.pretrained_model = pretrained_model
		return pretrained_model, loss_hist, train_acc_hist, val_acc_hist



	################################################################################
	####################[ Localization ]############################################
	################################################################################

	def init_transfer_convnet(self, pretrained_model=None):
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
		C, H, W = self.params['shape_loc']
		F1, F2, F3, FC = self.params['num_filters_loc']
		filter_sizes = self.params['filter_sizes_loc']
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

		# model['W3'] = np.random.randn(H * W * F2 / 4, FC) #fully connected layer
		model['W3'] = np.random.randn(F3, F2, filter_sizes[2], filter_sizes[2])
		model['b3'] = np.random.randn(F3)

		model['W4'] = np.random.randn(F3*(H/(2*2*2))*(W/(2*2*2)), FC)
		model['b4'] = np.random.randn(FC)

		model['W5'] = np.random.randn(FC, self.params['size_out'])
		model['b5'] = np.random.randn(self.params['size_out'])

		#=====[ Step 2: Scale weights	]=====
		regs = [2, 3, 4, 5]
		if pretrained_model:
			regs.insert(0, 1)
		for i in regs:
			model['W%d' % i] *= self.params['weight_scale']
			model['b%d' % i] *= self.params['bias_scale']

		#=====[ Step 3: Convert to dtype	]=====
		for k in model:
			model[k] = model[k].astype(self.params['dtype'], copy=False)

		return model


	def euclidean_loss(self, y_pred, y):
		"""returns summed loss, gradients for each sample"""
		assert y_pred.shape == y.shape

		#=====[ Step 1: Loss ]=====
		diffs = y_pred - y
		loss = np.mean(np.sqrt(np.sum(diffs**2, axis=1))) / 2

		#=====[ Step 2: Gradients ]=====
		grads = diffs
		return loss, grads


	def transfer_convnet(self, X, model, y=None, reg=0.0, dropout=None):
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
		conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
		conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
		conv_param_3 = {'stride': 1, 'pad': (W3.shape[2] - 1) / 2}
		pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
		dropout_param = {'p': dropout}
		dropout_param['mode'] = 'test' if y is None else 'train'

		#=====[ Step 2: Forward pass	]=====
		a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
		a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
		a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param_3, pool_param)
		a4, cache4 = affine_relu_forward(a3, W4, b4)
		scores, cache5 = affine_forward(a4, W5, b5)

		#=====[ Step 3: Return scores during feedforward inference	]=====
		if y is None:
			return scores

		#=====[ Step 4: Backward pass - loss and gradients	]=====
		data_loss, dscores = self.euclidean_loss(scores, y)
		da4, dW5, db5 = affine_backward(dscores, cache5)
		da3, dW4, db4 = affine_relu_backward(da4, cache4)
		da2, dW3, db3 = conv_relu_pool_backward(da3, cache3)
		da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
		dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)



		#=====[ Step 5: regularize gradients, get total loss	]=====
		grads = { 
					'W1': dW1, 'b1': db1, 
					'W2': dW2, 'b2': db2, 
					'W3': dW3, 'b3': db3,
					'W4': dW4, 'b4': db4,
					'W5': dW5, 'b5': db5
				}
		reg_loss = 0.0
		for p in ['W1', 'W2', 'W3', 'W4', 'W5']:
			W = model[p]
			reg_loss += 0.5 * reg * np.sum(W * W)
			grads[p] += reg * W
		loss = data_loss + reg_loss

		return loss, grads



	def train(self, X_train, y_train, X_val, y_val, pretrain=True,	
													dropout=None, reg=0.0001, 
													learning_rate=0.0007, 
													batch_size=100, num_epochs=100,
													learning_rate_decay=0.95, 
													update='rmsprop', verbose=True):
		"""
			trains localization convnet. 

			returns model, loss_hist, train_acc_hist, val_acc_hist
		"""
		trainer = EuclideanConvNetTrainer()
		
		if pretrain:
			model = self.init_transfer_convnet(self.pretrained_model)
		else:
			model = self.init_transfer_convnet()

		model, loss_hist, train_acc_hist, val_acc_hist = trainer.train(
																		X_train, y_train, 
																		X_val, y_val, 
																		model, self.transfer_convnet, 
																		dropout=dropout, reg=reg, 
																		learning_rate=learning_rate, 
																		batch_size=batch_size, num_epochs=num_epochs,
																		learning_rate_decay=learning_rate_decay, 
																		update=update, verbose=verbose
																	)
		self.model = model
		return model, loss_hist, train_acc_hist, val_acc_hist



	def predict(self, X):
		"""X -> coordinates of cube"""
		return self.transfer_convnet(X, self.model)


	def save(self, path):
		"""saves self to a path"""
		pickle.dump((self.model, self.pretrained_model, self.params), open(path, 'w'))


	def load(self, path):
		"""loads from a path"""
		self.model, self.pretrained_model, self.params = pickle.load(open(path))














