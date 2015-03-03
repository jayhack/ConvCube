import numpy as np
from scipy.misc import imresize
import cv2
from sklearn.pipeline import Pipeline

def resize(image, shape=(360, 640)):
	"""image -> smaller image. idempotent."""
	if not (image.shape[:2] == shape):			
		image = imresize(image, shape)
	return image


def imagenet_resize(image):
	"""image -> (45,80,3) shaped-image"""
	return resize(image, (45,80))


def grayscale(image):
	"""image -> grayscale image. idempotent."""
	if not len(image.shape) == 2:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return image


def resize_grayscale(image):
	"""image -> smaller, grayscale image. idempotent."""
	return resize(grayscale(image))