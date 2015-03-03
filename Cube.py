import numpy as np 
import scipy as sp
import cv2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

from preprocess import resize, grayscale, resize_grayscale
from keypoints import interesting_points, sift_descriptors


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







