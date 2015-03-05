import numpy as np
import scipy as sp
from cube import Cube
from convcube.localization import get_X_localization
from convcube.localization import LocalizationConvNet
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
	def __init__(self, loc_model=None):
		# super(ConvNetCube, self).__init__()
		self.loc_model = loc_model


	################################################################################
	####################[ CovNet: Localization ]####################################
	################################################################################

	def localize(self, image):
		"""frame -> coordinates of cube as tuple"""
		X = get_X_localization(image)
		convnet = LocalizationConvNet()
		return convnet.localization_convnet(X, self.loc_model)



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, image):
		"""frame -> disp_image with output drawn on it"""
		#=====[ Step 1: get localization	]=====
		coords = self.localize(image)

		#=====[ Step 2: scale coords	]=====
		coords[0] *= image.shape[0]
		coords[1] *= image.shape[1]
		disp_img = draw_points(image, coords, labels=[1])
		return disp_img