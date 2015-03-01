import numpy as np 
import scipy as sp
import skimage
from sklearn.pipeline import Pipeline

from preprocess import resize, grayscale, resize_grayscale
from image_points import interesting_points, sift_descriptors


class Cv2Cube(object):
	"""
	Class: Cv2Cube
	--------------
	Uses primarily cv2 and scipy in order to find, track and 
	interpret the cube 
	"""
	def __init__(self, classifier=None):
		super(Cv2Cube, self).__init__()
		self.classifier = classifier


	def get_keypoints(self, image):
		"""image -> keypoints as tuples"""
		return interesting_points(grayscale(resize(image)))


	def get_descriptors(self, image, keypoints):
		"""keypoints as tuples -> labelled corners"""
		return sift_descriptors(image, keypoints)


	def draw_keypoints(self, image, keypoints, labels=None):
		"""(image, keypoints as tuples, labels) -> image with keypoints drawn on it"""
		if labels is None:
			labels = [None]*len(keypoints)
		disp_image = image.copy()
		positive = [k for k,l in zip(keypoints, labels) if l]
		for p in positive:
			cv2.circle(disp_image, p, 5, color=(0, 255, 0), thickness=2)
		negative = [k for k,l in zip(keypoints, labels) if not l]
		for n in negative:
			cv2.circle(disp_image, n, 3, color=(0, 0, 255), thickness=1)
		return disp_image


	def find_cube(self, image):
		"""finds location from image"""
		raise NotImplementedError


	def update(self, image):
		""" """
		raise NotImplementedError







