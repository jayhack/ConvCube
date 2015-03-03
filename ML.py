import numpy as np
import cv2
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def construct_kpts_label_image(image, kpts):
    """(image, keypoints) -> image containing corners"""
    kpts = np.array(kpts).astype(np.uint16)
    label_image = np.zeros((image.shape[0], image.shape[1]))
    label_image[kpts[:,1], kpts[:,0]] = 1
    label_image = cv2.GaussianBlur(label_image, (5,5), 2)
    return label_image


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


def init_four_layer_convnet(  input_shape=(3, 64, 64), num_classes=100,
                            filter_sizes=(5, 5), num_filters=(16, 16, 16),
                            weight_scale=1e-2, bias_scale=0, dtype=np.float32):
    """
    Initialize a five-layer convnet with the following architecture:

    [conv - relu - pool] x 2 - affine - relu - dropout - affine - softmax

    Each pooling region is 2x2 stride 2 and each convolution uses enough padding
    so that all convolutions are "same".

    Inputs:
    - Input shape: A tuple (C, H, W) giving the shape of each input that will be
      passed to the ConvNet. Default is (3, 64, 64) which corresponds to
      TinyImageNet.
    - num_classes: Number of classes over which classification will be performed.
      Default is 100 for TinyImageNet-100-A / TinyImageNet-100-B.
    - filter_sizes: Tuple of 3 integers giving the size of the filters for the
      three convolutional layers. Default is (5, 5, 5) which corresponds to 5x5
      filter at each layer.
    - num_filters: Tuple of 4 integers where the first 3 give the number of
      convolutional filters for the three convolutional layers, and the last
      gives the number of output neurons for the first affine layer.
      Default is (32, 32, 64, 128).
    - weight_scale: All weights will be randomly initialized from a Gaussian
      distribution whose standard deviation is weight_scale.
    - bias_scale: All biases will be randomly initialized from a Gaussian
      distribution whose standard deviation is bias_scale.
    - dtype: numpy datatype which will be used for this network. Float32 is
      recommended as it will make floating point operations faster.
    """
    C, H, W = input_shape
    F1, F2, FC = num_filters
    model = {}
    model['W1'] = np.random.randn(F1, 3, filter_sizes[0], filter_sizes[0])
    model['b1'] = np.random.randn(F1)
    model['W2'] = np.random.randn(F2, F1, filter_sizes[1], filter_sizes[1])
    model['b2'] = np.random.randn(F2)
    model['W3'] = np.random.randn(H * W * F2 / 64, FC)
    model['b3'] = np.random.randn(FC)
    model['W4'] = np.random.randn(FC, num_classes)
    model['b4'] = np.random.randn(num_classes)

    for i in [1, 2, 3, 4]:
        model['W%d' % i] *= weight_scale
        model['b%d' % i] *= bias_scale

    for k in model:
        model[k] = model[k].astype(dtype, copy=False)

    return model



def four_layer_convnet(X, model, y=None, reg=0.0, dropout=1.0,
                       extract_features=False, compute_dX=False,
                       return_probs=False):
  """
  Compute the loss and gradient for a five layer convnet with the architecture

  [conv - relu - pool] x 2 - affine - relu - dropout - affine - softmax

  Each conv is stride 1 with padding chosen so the convolutions are "same";
  all padding is 2x2 stride 2.

  We use L2 regularization on all weight matrices and no regularization on
  biases.

  This function can output several different things:

  If y not given, then this function will output extracted features,
  classification scores, or classification probabilities depending on the
  values of the extract_features and return_probs flags.

  If y is given, then this function will output either (loss, gradients)
  or dX, depending on the value of the compute_dX flag.

  Inputs:
  - X: Input data of shape (N, C, H, W)
  - model: Dictionary mapping string names to model parameters. We expect the
    following parameters:
    W1, b1, W2, b2, W3, b3: Weights and biases for the conv layers
    W4, b4, W5, b5: Weights and biases for the affine layers
  - y: Integer vector of shape (N,) giving labels for the data points in X.
    If this is given then we will return one of (loss, gradient) or dX;
    If this is not given then we will return either class scores or class
    probabilities.
  - reg: Scalar value giving the strength of L2 regularization.
  - dropout: The probability of keeping a neuron in the dropout layer

  Outputs:
  This function can return several different things, depending on its inputs
  as described above.

  If y is None and extract_features is True, returns:
  - features: (N, H) array of features, where H is the number of neurons in the
    first affine layer.
  
  If y is None and return_probs is True, returns:
  - probs: (N, L) array of normalized class probabilities, where probs[i][j]
    is the probability that X[i] has label j.

  If y is None and return_probs is False, returns:
  - scores: (N, L) array of unnormalized class scores, where scores[i][j] is
    the score assigned to X[i] having label j.

  If y is not None and compute_dX is False, returns:
  - (loss, grads) where loss is a scalar value giving the loss and grads is a
    dictionary mapping parameter names to arrays giving the gradient of the
    loss with respect to each parameter.

  If y is not None and compute_dX is True, returns:
  - dX: Array of shape (N, C, H, W) giving the gradient of the loss with
    respect to the input data.
  """
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']
  W4, b4 = model['W4'], model['b4']

  conv_param_1 = {'stride': 1, 'pad': (W1.shape[2] - 1) / 2}
  conv_param_2 = {'stride': 1, 'pad': (W2.shape[2] - 1) / 2}
  pool_param = {'stride': 2, 'pool_height': 2, 'pool_width': 2}
  dropout_param = {'p': dropout}
  dropout_param['mode'] = 'test' if y is None else 'train'

  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param_1, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param_2, pool_param)
  a3, cache3 = affine_relu_forward(a2, W3, b3)
  d3, cache4 = dropout_forward(a3, dropout_param)
  scores, cache5 = affine_forward(d3, W4, b4)

  if y is None:
    if return_probs:
      probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
      probs /= np.sum(probs, axis=1, keepdims=True)
      return probs
    else:
      return scores

  data_loss, dscores = softmax_loss(scores, y)
  dd3, dW4, db4 = affine_backward(dscores, cache5)
  da3 = dropout_backward(dd3, cache4)
  da2, dW3, db3 = affine_relu_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)

  if compute_dX:
    ###########################################################################
    # TODO: Return the gradient of the loss with respect to the input.        #
    # HINT: This should be VERY simple!                                       #
    ###########################################################################
    return dX
    ###########################################################################
    #                         END OF YOUR CODE                                #  
    ###########################################################################

  grads = {
    'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
    'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4
  }

  reg_loss = 0.0
  for p in ['W1', 'W2', 'W3', 'W4']:
    W = model[p]
    reg_loss += 0.5 * reg * np.sum(W)
    grads[p] += reg * W
  loss = data_loss + reg_loss

  return loss, grads
