import numpy as np
import cv2
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def make_output_coords(kpts):
    """(image, kpts) -> output of center of cube
        should be paired with a linear top layer

        (mean of the keypoints)
    """
    kpts = np.array(kpts) / 8.0
    return np.mean(kpts, axis=0)


def make_output_heatmap(image, kpts):
    """(image, kpts) -> output of map"""
    kpts = np.array(kpts).astype(np.uint16)
    label_image = np.zeros((image.shape[0], image.shape[1]))
    label_image[kpts[:,1], kpts[:,0]] = 1
    label_image = cv2.GaussianBlur(label_image, (5,5), 2)
    return label_image


def get_convnet_inputs(frame):
  """ModalDB.Frame -> (downsized image, corner heatmap, coords)"""
  raise NotImplementedError
  

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

