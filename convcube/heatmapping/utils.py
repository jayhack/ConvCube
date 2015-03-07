import numpy as np
import cv2
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
from convcube.cv.preprocess import resize
from convcube.cv.preprocess import put_channels_first
from convcube.cv.keypoints import denormalize_points
from convcube.cv.keypoints import tuples2array
from convcube.cv.cropping import crop_image
from convcube.cv.cropping import crop_points
from convcube.utils.wrangling import iter_labeled_frames


def heatmap_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46, 80))
	return image


def get_X_heatmap(image, box):
	"""image -> array for input to convnet"""
	box = denormalize_points(box, image)
	if (box[1][0] - box[0][0] > 5) and (box[1][1] - box[0][1] > 5):
		image = crop_image(image, box)
		return put_channels_first(heatmap_resize(image))
	else:
		return None


def get_y_heatmap(image, kpts, box):
	"""list of interior corners as tuples -> center of cube

		TODO: y_localization to portion across screen
		TODO: change output to (center, size)
	"""
	try:
		if kpts is None or kpts == []:
			return None

		box = denormalize_points(box, image)
		tl, br = box[0], box[1]

		y = np.zeros((int(br[1])-int(tl[1]), int(br[0])-int(tl[0])))

		if not y.shape[0] > 0 or not y.shape[1] > 0:
			return None

		kpts = denormalize_points(kpts, image)
		kpts = crop_points(kpts, box)
		if len(kpts) == 0:
			return None

		kpts = tuples2array(kpts).astype(np.uint16)
		y[kpts[:,1], kpts[:,0]] = 1

		kernel = np.ones((5,5),np.uint8)
		y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)
		y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)
		# y = cv2.GaussianBlur(y, (5,5), 3)
		y = y.astype(np.float32)
		y = y / y.max()
		y = heatmap_resize(y)

		return heatmap_resize(y).flatten()
	except:
		return None


def get_convnet_inputs_heatmap(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_heatmap(frame['image'], frame['bounding_box'])
	y = get_y_heatmap(frame['image'], frame['interior_points'], frame['bounding_box'])
	return X, y


def load_dataset_heatmap(client, train_size=0.9):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []
	for frame in iter_labeled_frames(client):
		X_, y_ = get_convnet_inputs_heatmap(frame)
		if not X_ is None and not y_ is None:
			
			y_ = y_.astype(np.float32)
			y_ /= y_.max()
			X.append(X_)
			y.append(y_)

	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)
