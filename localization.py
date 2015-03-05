import numpy as np
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *
from preprocess import resize
from preprocess import put_channels_first
from wrangling import iter_labeled_frames
from sklearn.cross_validation import train_test_split


################################################################################
####################[ Formatting/Loading Data ]#################################
################################################################################

def localization_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46,80)) 
	return image


def get_X_localization(image):
	"""image -> array for input to convnet"""
	return put_channels_first(localization_resize(image))


def get_y_localization(kpts):
	"""list of interior corners as tuples -> center of cube

		TODO: change output to (center, size)
	"""
	return np.mean(np.array(kpts), axis=0)


def get_convnet_inputs_localization(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_localization(frame['image'])
	y = get_y_localization(frame['interior_points'])
	return X, y


def load_dataset_localization(client, train_size=0.75):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []
	for frame in iter_labeled_frames(client):
		X_, y_ = get_convnet_inputs_localization(frame)
		X.append(X_)
		y.append(y_)
	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)





################################################################################
####################[ ConvNet Init/Compute ]####################################
################################################################################

def init_localization_convnet(  pretrained_model,
                                input_shape=(3, 46, 80), 
                                filter_size=5, 
                                num_filters=(16, 32),
                                weight_scale=1e-2, 
                                bias_scale=0, 
                                dtype=np.float32):
    """
    Initialize a three layer ConvNet with the following architecture:

    conv - relu - pool - affine - relu - dropout - affine - euclidean_loss

    (outputs coordinates in R2 corresponding to Cube location)

    The convolutional layer uses stride 1 and has padding to perform "same"
    convolution, and the pooling layer is 2x2 stride 2.

    Inputs:
    - input_shape: Tuple (C, H, W) giving the shape of each training sample.
    Default is (3, 32, 32) for CIFAR-10.
    - num_classes: Number of classes over which classification will be performed.
    Default is 10 for CIFAR-10.
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
    F1, FC = num_filters
    model = {}
    model['W1'] = pretrained_model['W1']
    model['b1'] = pretrained_model['b1']
    model['W2'] = np.random.randn(H * W * F1 / 4, FC)
    model['b2'] = np.random.randn(FC)
    model['W3'] = np.random.randn(FC, 2)
    model['b3'] = np.random.randn(2)

    for i in [2, 3]:
        model['W%d' % i] *= weight_scale
        model['b%d' % i] *= bias_scale

    for k in model:
        model[k] = model[k].astype(dtype, copy=False)

    return model


def euclidean_loss(y_pred, y):
    """return euclidean loss"""
    assert y_pred.shape == y.shape
    
    #=====[ Step 1: loss ]=====
    diffs = y_pred - y
    loss = np.sum(np.sum(diffs**2, axis=1))

    #=====[ Step 2: gradients ]=====
    grads = 2 * diffs
    return loss, grads



def localization_convnet(X, model, y=None, reg=0.0, dropout=None):
  """
  Compute the loss and gradient for a simple three layer ConvNet that uses
  the following architecture:

  conv - relu - pool - affine - relu - dropout - affine - 

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
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']

  conv_param = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
  pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = affine_relu_forward(a1, W2, b2)
  if dropout is None:
    scores, cache4 = affine_forward(a2, W3, b3)
  else:
    d2, cache3 = dropout_forward(a2, dropout_param)
    scores, cache4 = affine_forward(d2, W3, b3)

  if y is None:
    #====[ this is just the euclidean coordinates... ]=====
    return scores

  data_loss, dscores = euclidean_loss(scores, y)
  if dropout is None:
    da2, dW3, db3 = affine_backward(dscores, cache4)
  else:
    dd2, dW3, db3 = affine_backward(dscores, cache4)
    da2 = dropout_backward(dd2, cache3)
  da1, dW2, db2 = affine_relu_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  grads = { 'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3 }

  reg_loss = 0.0
  for p in ['W1', 'W2', 'W3']:
    W = model[p]
    reg_loss += 0.5 * reg * np.sum(W * W)
    grads[p] += reg * W
  loss = data_loss + reg_loss

  return loss, grads




################################################################################
####################[ ConvNet Train ]###########################################
################################################################################

class LocalizationConvNetTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, y, X_val, y_val, 
            model, loss_function, 
            reg=0.0, dropout=1.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            update='momentum', sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            augment_fn=None, predict_fn=None,
            verbose=False):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of training data; each X[i] is a training sample.
    - y: Vector of training labels; y[i] gives the label for X[i].
    - X_val: Array of validation data
    - y_val: Vector of validation labels
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - dropout: Amount of dropout to use. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
    - num_epochs: The number of epochs to take over the training data.
    - batch_size: The number of training samples to use at each iteration.
    - acc_frequency: If set to an integer, we compute the training and
      validation set error after every acc_frequency iterations.
    - augment_fn: A function to perform data augmentation. If this is not
      None, then during training each minibatch will be passed through this
      before being passed as input to the network.
    - predict_fn: A function to mutate data at prediction time. If this is not
      None, then during each testing each minibatch will be passed through this
      before being passed as input to the network.
    - verbose: If True, print status after each epoch.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """

    N = X.shape[0]

    #=====[ Step 1: setup ]=====
    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    #==========[ ITERATIONS ]==========
    for it in xrange(num_iters):
      if it % 10 == 0:  print 'starting iteration ', it

      #=====[ Get batch of data ]=====
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y


      #=====[ PERFORM DATA AUGMENTATION ]=====
      if augment_fn is not None:
        X_batch = augment_fn(X_batch)

      #=====[ EVALUATE COST AND GRADIENT ]=====
      cost, grads = loss_function(X_batch, model, y_batch, reg=reg, dropout=dropout)
      loss_history.append(cost)

      #=====[ PARAMETER UPDATE ]=====
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          dx = momentum * self.step_cache[p] - learning_rate * grads[p]
          self.step_cache[p] = dx
        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          self.step_cache[p] = self.step_cache[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + 1e-8)
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx



      #=====[ EVALUATION ON VALIDATION SET ]=====
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y

        #=====[ FORWARD PASS ON TRAINING SET]=====
        # Computing a forward pass with a batch size of 1000 will is no good,
        # so we batch it
        y_pred_train = []
        train_acc = []
        for i in xrange(X_train_subset.shape[0] / 10):
          X_train_slice = X_train_subset[i*10:(i+1)*10]
          if predict_fn is not None:
            X_train_slice = predict_fn(X_train_slice)
          
          scores = loss_function(X_train_slice, model) #(batch_size x 2) array of predicted coordinates
          y_pred_train.append(scores)

        y_pred_train = np.vstack(y_pred_train)
        diffs = y_pred_train - y_train_subset[:y_pred_train.shape[0]]
        train_acc = float(np.mean(np.sum(diffs**2, axis=1)))
        train_acc_history.append(train_acc)

        #=====[ FORWARD PASS ON VALIDATION SET ]=====
        # evaluate val accuracy, but split the validation set into batches
        y_pred_val = []
        for i in xrange(X_val.shape[0] / 10):
          X_val_slice = X_val[i*10:(i+1)*10]
          if predict_fn is not None:
            X_val_slice = predict_fn(X_val_slice)

          scores = loss_function(X_val_slice, model)
          y_pred_val.append(scores)

        y_pred_val = np.vstack(y_pred_val)
        diffs = y_pred_val - y_val[:y_pred_val.shape[0]]
        val_acc = float(np.mean(np.sum(diffs**2, axis=1)))
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history



