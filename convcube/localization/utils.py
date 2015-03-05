import numpy as np
from sklearn.cross_validation import train_test_split
from convcube.cv.preprocess import resize
from convcube.cv.preprocess import put_channels_first
from convcube.utils.wrangling import iter_labeled_frames


def localization_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46, 80))
	return image


def get_X_localization(image):
	"""image -> array for input to convnet"""
	return put_channels_first(localization_resize(image))


def get_y_localization(kpts):
	"""list of interior corners as tuples -> center of cube

		TODO: y_localization to portion across screen
		TODO: change output to (center, size)
	"""
	kpts = np.array(kpts).astype(np.float32)
	kpts[0] /= 360.0
	kpts[1] /= 640.0 
	return np.mean(np.array(kpts), axis=0)


def get_convnet_inputs_localization(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_localization(frame['image'])
	y = get_y_localization(frame['interior_points'])
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
