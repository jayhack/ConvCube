import numpy as np
import scipy as sp
from cube import Cube
from convcube.localization import get_X_localization
from convcube.localization import localization_convnet
from convcube.drawing import draw_points

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
		super(ConvNetCube, self).__init__()
		self.loc_convnet = loc_convnet


	################################################################################
	####################[ CovNet: Localization ]####################################
	################################################################################

	def localize(self, image):
		"""frame -> coordinates of cube as tuple"""
		X = image2localization_input(frame)
		coords = localization_convnet(self.loc_convnet, X)



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, frame):
		"""frame -> disp_image with output drawn on it"""
		coords = self.localize(frame)
		disp_img = draw_points(frame, coords, labels=[1])
		return disp_img