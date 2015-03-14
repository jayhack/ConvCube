import numpy as np
import scipy as sp
import cv2
from cube import Cube
from convcube.convnets.localization import get_X_localization
from convcube.convnets.pinpointing import cropped_to_X_pinpointing
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
	def __init__(self, loc_convnet=None, pin_convnet=None):
		# super(ConvNetCube, self).__init__()
		self.loc_convnet = loc_convnet
		self.pin_convnet = pin_convnet


	################################################################################
	####################[ CovNet: Localization ]####################################
	################################################################################

	def localize(self, image):
		"""image -> tl, br of bounding box around cube (normalized)"""
		if self.loc_convnet is None:
			raise AttributeError("You need to load a localization convnet")

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


	def draw_localization(self, image, tl=None, br=None):
		"""(image, tl?, br?) -> bounding box around localization"""
		if tl is None or br is None:
			tl, br = self.localize(image)

		assert type(tl) == tuple and type(br) == tuple

		tl, br = denormalize_points([tl, br], image)
		disp_img = image.copy()
		cv2.rectangle(disp_img, tl, br, (0, 255, 0), thickness=2)
		return disp_img


	################################################################################
	####################[ CovNet: PinPointing ]#####################################
	################################################################################

	def pinpoint(self, image):
		"""(cropped) image -> prediction of location of center(s) of sides"""
		if self.pin_convnet is None:
			raise AttributeError("You need to load a pinpointing convnet")

		X = cropped_to_X_pinpointing(image)
		y_pred = self.pin_convnet.predict(X)[0]
		x, y = y_pred[0], y_pred[1]
		return x, y


	def draw_pinpoint(self, image, pt=None):
		"""(image, pt?) -> point around pinpoint"""
		if pt is None:
			pt = self.pinpoint(image)

		pt = denormalize_points([pt], image)[0]
		x,y = pt[0], pt[1]
		disp_img = image.copy()
		disp_img[y-2:y+2, x-2:x+2] = (0, 255, 0)
		return disp_img


	################################################################################
	####################[ CovNet: PinPointing ]#####################################
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