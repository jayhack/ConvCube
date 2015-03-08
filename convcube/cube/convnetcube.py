import numpy as np
import scipy as sp
import cv2
from cube import Cube
from convcube.convnets import get_X_localization
from convcube.convnets import get_X_pinpointing
from convcube.convnets import EuclideanConvNet
from convcube.utils import draw_points

class ConvNetCube(Cube):
	"""
	Class: ConvNetCube
	------------------
	Uses primarily cv2 and scipy in order to find, track and 
	interpret the cube 

	Args:
	-----
	- loc_convnet: convnet trained for localization. That is, image ->
	  approximate coordinates of rubiks cube in image.
	  TODO: also find scale!
	"""
	def __init__(self, loc_convnet=None):
		# super(ConvNetCube, self).__init__()
		self.loc_convnet = loc_convnet


	################################################################################
	####################[ CovNet: Localization ]####################################
	################################################################################

	def localize(self, image):
		"""frame -> coordinates of cube as tuple"""
		X = get_X_localization(image)
		y = self.loc_convnet.predict(X)[0]
		y = y.reshape((2,2))
		y[:,0] *= 640.0
		y[:,1] *= 360.0
		y = y.astype(np.uint16)

		tl = (y[0,0], y[0,1])
		br = (y[1,0], y[1,1]) #(x,y) tuples

		return tl, br



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, image):
		"""frame -> disp_image with output drawn on it"""

		#=====[ Step 1: get localization	]=====
		tl, br = self.localize(image)


		#=====[ Step 2: scale coords	]=====
		disp_img = image.copy()
		cv2.rectangle(disp_img, tl, br, (0, 255, 0), thickness=2)
		# disp_img = draw_points(image, coords, labels=[True])
		return disp_img