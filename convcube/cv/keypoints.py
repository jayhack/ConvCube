import numpy as np
import cv2
from preprocess import resize, grayscale, resize_grayscale


################################################################################
####################[ Point Conversion Utilities ]##############################
################################################################################

def array2tuples(pts):
	"""np.array -> list of tuples"""
	return [tuple(p) for p in pts.tolist()]


def tuples2array(tuples):
	"""list of tuples -> np.array"""
	return np.array(tuples)


def kpts2tuples(kpts):
	"""list of cv2.KeyPoints -> list of (x,y)"""
	return [k.pt for k in kpts]


def tuples2kpts(tuples, size=5):
	"""list of (x,y) -> list of cv2.KeyPoints"""
	return [cv2.KeyPoint(t[0], t[1], size) for t in tuples]


def tuples2ints(tuples):
	"""converts list of (x,y) to all ints"""
	return [(int(p[0]), int(p[1])) for p in tuples]


def normalize_points(pts, image):
	"""(pts as tuples, image) -> normalized points"""
	pts = tuples2array(pts).astype(np.float32)
	H, W = image.shape[0], image.shape[1]
	pts[:,0] /= float(W)
	pts[:,1] /= float(H)
	return array2tuples(pts)


def denormalize_points(pts, image):
	"""(pts as tuples, image) -> denormalized points"""
	pts = tuples2array(pts).astype(np.float32)
	H, W = image.shape[0], image.shape[1]
	pts[:,0] *= float(W)
	pts[:,1] *= float(H)
	pts = pts.astype(np.uint16)
	return array2tuples(pts)





################################################################################
####################[ Keypoint Extraction ]#####################################
################################################################################

def harris_corners(image):
	"""(grayscale) image -> harris corners as a list of tuples"""
	image = grayscale(image)
	dst = cv2.cornerHarris(image, 5, 5, 0.04)
	corners_img = (dst>0.01*dst.max())
	ys, xs = np.nonzero(corners_img)
	return zip(xs, ys)


def interesting_points(image):
	"""(grayscale) gray image -> interesting points as list of tuples"""
	image = grayscale(image)
	feature_params = {
						'maxCorners':100,
						'qualityLevel':0.01,
						'minDistance':5,
						'blockSize':3
					}
	arr = cv2.goodFeaturesToTrack(image, mask = None, **feature_params)
	return array2tuples(arr[:, 0, :].astype(np.uint16))


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


def kpts_to_image_crop(image, kpts):
	"""(image, kpts) -> tl, br bounds on image crop"""
	kpts = np.array(kpts)
	center = np.mean(kpts, axis=0)
	sides = np.max(kpts, axis=0) - np.min(kpts, axis=0)
	tl = center - (1.3*sides)
	br = center + (1.3*sides)
	tl[tl < 0] = 0
	br[br < 0] = 0

	if tl[0] > image.shape[1]:
		tl[0] = image.shape[1]
	if br[0] > image.shape[1]:
		br[0] = image.shape[1]

	if tl[1] > image.shape[0]:
		tl[1] = image.shape[0]
	if br[1] > image.shape[0]:
		br[1] = image.shape[0]

	return (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1]))

