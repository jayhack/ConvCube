import numpy as np
import cv2
from preprocess import resize, grayscale, resize_grayscale


def kpts_to_tuples(kpts):
	"""list of cv2.KeyPoints -> list of (x,y)"""
	return [k.pt for k in kpts]


def tuples_to_kpts(tuples, size=5):
	"""list of (x,y) -> list of cv2.KeyPoints"""
	return [cv2.KeyPoint(t[0], t[1], size) for t in tuples]


def array_to_tuples(array):
	"""np.array -> list of (x,y)"""
	return [tuple(p) for p in array[:, 0, :]]


def harris_corners(image):
	"""(grayscale) image -> harris corners as a list of tuples"""
	image = grayscale(image)
	dst = cv2.cornerHarris(image, 7, 5, 0.04)
	corners_img = (dst>0.01*dst.max())
	ys, xs = np.nonzero(corners_img)
	return zip(xs, ys)


def interesting_points(image):
	"""(grayscale) gray image -> interesting points as list of tuples"""
	image = grayscale(image)
	feature_params = {
						'maxCorners':1000,
						'qualityLevel':0.01,
						'minDistance':8,
						'blockSize':5
					}
	arr = cv2.goodFeaturesToTrack(image, mask = None, **feature_params)
	return array_to_tuples(arr)


def sift_points(gray):
	"""(grayscale) image -> sift keypoints as a list of tuples"""
	sift = cv2.SIFT()
	points = sift.detect(gray, mask=None)
	return kpts_to_tuples(points)


def sift_descriptors(image, tuples):
	"""((grayscale) image, points as tuples) -> np.array of SIFT descriptors"""
	gray = resize_grayscale(image)
	sift = cv2.SIFT()
	kp, desc = sift.compute(gray, tuples_to_kpts(tuples))
	return desc
