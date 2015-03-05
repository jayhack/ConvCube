import numpy as np 
import scipy as sp
import cv2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

from preprocess import resize
from preprocess import grayscale
from preprocess import resize_grayscale
from preprocess import put_channels_first
from preprocess import localization_resize
from preprocess import make_convnet_input
from keypoints import interesting_points
from keypoints import sift_descriptors
from ML import transfer_convnet
from drawing import draw_points


class Cv2Cube(object):
	"""
	Class: Cv2Cube
	--------------
	Uses primarily cv2 and scipy in order to find, track and 
	interpret the cube 
	"""
	def __init__(self, classifier=None):
		super(Cv2Cube, self).__init__()
		self.clf = classifier




	################################################################################
	####################[ Feature Extraction ]######################################
	################################################################################

	def get_descriptors(self, image, kpts):
		"""(image, keypoints as tuples) -> descriptors"""
		return sift_descriptors(image, kpts)


	def get_kpts_descriptors(self, image):
		"""image -> (keypoints as tuples, descriptors)"""
		image = resize_grayscale(image)
		kpts = interesting_points(image)
		desc = sift_descriptors(image, kpts)
		return kpts, desc


	def draw_kpts(self, image, keypoints, labels=None):
		"""(image, keypoints as tuples, labels) -> image with keypoints drawn on it"""
		if labels is None:
			labels = [None]*len(keypoints)
		disp_image = image.copy()
		negative = [k for k,l in zip(keypoints, labels) if not l]
		positive = [k for k,l in zip(keypoints, labels) if l]
		for n in negative:
			cv2.circle(disp_image, n, 3, color=(0, 0, 255), thickness=1)
		for p in positive:
			cv2.circle(disp_image, p, 5, color=(0, 255, 0), thickness=2)
		return disp_image





	################################################################################
	####################[ Classification ]##########################################
	################################################################################

	def classify_kpts(self, desc):
		"""(keypoints as tuples, descriptors) -> labels"""
		return self.clf.predict(desc)


	def train_kpts_classifier(self, X, y):
		"""(desc, labels) -> keypoints classifier"""
		self.clf.fit(X,y)
		return self.clf


	def evaluate_kpts_classifier(self, X, y):
		"""(desc, labels) -> scores in cross-validation of self.clf"""
		return cross_val_score(self.clf, X, y)






	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, image):
		"""image -> disp_image with output drawn on it"""
		kpts, desc = self.get_kpts_descriptors(image)
		labels = self.classify_kpts(desc)
		disp_img = self.draw_kpts(image, kpts, labels)
		return disp_img


	def find_cube(self, image):
		"""finds location from image"""
		raise NotImplementedError


	def update(self, image):
		""" """
		raise NotImplementedError




class Cube(object):
	"""
	Class: Cube
	-----------
	Abstract class for CV on rubiks cube.
	"""
	def __init__(self):
		pass

	def load(self, path):
		"""loads classifiers"""
		raise NotImplementedError

	def save(self, path):
		"""saves consumed data"""
		raise NotImplementedError

	def update(self, frame):
		"""updates current state"""
		raise NotImplementedError

	def draw(self, frame):
		"""draws current state on frame"""
		raise NotImplementedError



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

	def preprocess_convnet(self, frame):
		"""frame -> input to convnet. idempotent"""
		return make_convnet_input(frame)


	def localize(self, frame):
		"""frame -> coordinates of cube as tuple"""
		X = self.preprocess_convnet(frame)
		coords = transfer_convnet(self.loc_convnet, X)



	################################################################################
	####################[ Online ]##################################################
	################################################################################

	def draw_output(self, frame):
		"""frame -> disp_image with output drawn on it"""
		coords = self.localize(frame)
		disp_img = draw_points(frame, coords, labels=[1])
		return disp_img


