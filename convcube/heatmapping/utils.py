import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from convcube.cv.preprocess import resize
from convcube.cv.preprocess import put_channels_first
from convcube.cv.keypoints import kpts_to_image_crop
from convcube.utils.wrangling import iter_labeled_frames


def heatmap_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46, 80))
	return image


def get_X_heatmap(image, kpts):
	"""image -> array for input to convnet"""
	tl, br = kpts_to_image_crop(image, kpts)
	extracted = image[tl[1]:br[1], tl[0]:br[0]]
	return heatmap_resize(extracted)


def get_y_heatmap(image, kpts):
	"""list of interior corners as tuples -> center of cube

		TODO: y_localization to portion across screen
		TODO: change output to (center, size)
	"""
	tl, br = kpts_to_image_crop(image, kpts)
	kpts = np.array(kpts)
	kpts[:,0] -= tl[0]
	kpts[:,1] -= tl[1]
	kpts = kpts.astype(np.uint16)

	y = np.zeros((br[0]-tl[0], br[1]-tl[1]))
	y[kpts[:,1], kpts[:,0]] = 1
	y = cv2.GaussianBlur(y, (7,7), 3)
	kernel = np.ones((10,10),np.uint8)
	y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)
	y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, kernel)
	y = cv2.GaussianBlur(y, (5,5), 3)
	return heatmap_resize(y)


def get_convnet_inputs_heatmap(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_heatmap(frame['image'], frame['interior_points'])
	y = get_y_heatmap(frame['image'], frame['interior_points'])
	return X, y


def load_dataset_heatmap(client, train_size=0.9):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []
	for frame in iter_labeled_frames(client):
		X_, y_ = get_convnet_inputs_heatmap(frame)
		X.append(X_)
		y.append(y_)
	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)
