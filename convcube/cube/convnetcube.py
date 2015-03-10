import numpy as np
import scipy as sp
import cv2
from cube import Cube
from convcube.convnets import get_X_localization
from convcube.convnets import get_X_pinpointing
from convcube.convnets import EuclideanConvNet
from convcube.cv.keypoints import array2tuples
from convcube.cv.keypoints import denormalize_points
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
		"""frame -> tl, br of bounding box around cube"""
		X = get_X_localization(image)
		y = self.loc_convnet.predict(X)[0]
		y = y.reshape((2,2))
		y = array2tuples(y)
		center = y[0]
		scale_x = y[1][0]
		scale_y = y[1][1]
		tl = (center[0] - scale_x/2, center[1] - scale_y/2)
		br = (center[0] + scale_x/2, center[1] + scale_y/2)
		return tl, br



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, image):
		"""frame -> disp_image with output drawn on it"""

		#=====[ Step 1: get localization	]=====
		tl, br = self.localize(image)
		tl, br = denormalize_points([tl, br], image)

		#=====[ Step 2: scale coords	]=====
		disp_img = image.copy()
		cv2.rectangle(disp_img, tl, br, (0, 255, 0), thickness=2)
		return disp_img