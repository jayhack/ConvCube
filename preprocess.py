import numpy as np
from scipy.misc import imresize
import cv2

class ResizeT(object):
	"""
	Transform: ResizeT
	------------------
	downsizes by 'scale'
	"""
	def __init__(self, scale=0.5):
		self.scale = scale

	def fit(self, data):
		pass

	def transform(self, data):
		img = data
		img = imresize(img, self.scale)
		return img


class HarrisCornersT(object):
	"""
	Transform: GetHarrisCornersT
	----------------------------
	color image -> harris corners as a list of tuples
	"""
	def __init__(self):
		pass

	def fit(self, data):
		pass

	def transform(self, data):
		gray = np.float32(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))
		dst = cv2.cornerHarris(gray, 7, 5, 0.04)
		corners_img = (dst>0.01*dst.max())
		ys, xs = np.nonzero(corners_img)
		return zip(xs, ys)


class InterestingPointsT(object):
	"""
	Transform: InterestingPointsT
	-----------------------------
	color image -> interesting points to track as a list 
	of tuples 
	"""
	def __init__(self):
		pass

	def fit(self, data):
		pass

	def transform(self, data):
		feature_params = {
							'maxCorners':1000,
							'qualityLevel':0.01,
							'minDistance':8,
							'blockSize':5
						}
		gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
		p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)
		p0 = p0[:, 0, :]
		return [tuple(p) for p in p0]


class SurfDescriptorT(object):
	"""
	Transform: SuftDetectorT
	------------------------
	list of points -> set of surf descriptors 
	"""
	def __init__(self):
		self.surf = cv2.SURF(400)

	def fit(self, data):
		pass

	def transform(self, data):
		image, keypoints = data
		return surf.compute(image, keypoints)

