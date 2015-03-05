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
		y[0] = (y[0] * 640).astype(np.uint16)
		y[1] = (y[1] * 360).astype(np.uint16)
		return tuple(y)



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, image):
		"""frame -> disp_image with output drawn on it"""

		#=====[ Step 1: get localization	]=====
		coords = self.localize(image)

		#=====[ Step 2: scale coords	]=====
		disp_img = draw_points(image, coords, labels=[True])
		return disp_img