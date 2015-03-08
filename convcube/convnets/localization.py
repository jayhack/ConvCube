import numpy as np
from sklearn.cross_validation import train_test_split
from convcube.cv.preprocess import resize
from convcube.cv.preprocess import put_channels_first
from convcube.cv.keypoints import kpts_to_image_crop
from convcube.utils.wrangling import iter_labeled_frames


def localization_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46, 80))
	return image


def get_X_localization(image):
	"""image -> array for input to convnet"""
	return put_channels_first(localization_resize(image))


def get_y_localization(box):
	"""list of interior corners as tuples -> center of cube

		TODO: y_localization to portion across screen
		TODO: change output to (center, size)
	"""
	y = np.array(box).reshape(1,4)
	# tl, br = kpts_to_image_crop(aimage, kpts)
	# tl = ((float(tl[0]) / 640.0), (float(tl[1]) / 360.0))
	# br = ((float(br[0]) / 640.0), (float(br[1]) / 360.0))
	# y = np.array([tl, br]).flatten()
	return y


def get_convnet_inputs_localization(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_localization(frame['image'])
	y = get_y_localization(frame['bounding_box'])
	return X, y


def load_dataset_localization(client, train_size=0.75):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []
	for frame in iter_labeled_frames(client):
		X_, y_ = get_convnet_inputs_localization(frame)
		X.append(X_)
		y.append(y_)
	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)
