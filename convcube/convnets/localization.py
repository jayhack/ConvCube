import random
import numpy as np
import itertools
from sklearn.cross_validation import train_test_split
from convcube.cv.preprocess import resize
from convcube.cv.preprocess import put_channels_first
from convcube.cv.keypoints import kpts_to_image_crop
from convcube.cv.keypoints import tuples2array
from convcube.utils.wrangling import iter_labeled_frames
from ModalDB import ModalClient, Video, Frame


def localization_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (64, 64):
		image = resize(image, (64, 64))
	return image


def get_X_localization(image):
	"""image -> array for input to convnet"""

	image = localization_resize(image).astype(np.float32)

	#=====[ Scale to -1, 1	]=====
	image *= (2.0/255.0)
	image -= 1.0

	# mean_pixel = image.mean(axis=0).mean(axis=0)
	# image = image - mean_pixel
	return put_channels_first(image)


def get_y_localization(box):
	"""list of interior corners as tuples -> center of cube

		TODO: y_localization to portion across screen
		TODO: change output to (center, size)
	"""
	points = tuples2array(box)
	center = points.mean(axis=0)
	scale = np.mean(abs(points - center)*2, axis=0)
	return np.hstack([center, scale]).reshape(1, 4)


def get_convnet_inputs_localization(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_localization(frame['image'])
	y = get_y_localization(frame['bounding_box'])
	return X, y


def load_dataset_localization(client, train_size=0.75):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	Xs, ys = [], []
	for video in client.iter(Video):
		X_video, y_video = [], []
		for frame in video.iter_children(Frame):
			if not frame['bounding_box'] is None and len(frame['bounding_box']) == 2:
				X, y = get_convnet_inputs_localization(frame)
				X_video.append(X)
				y_video.append(y)

		Xs.append(X_video)
		ys.append(y_video)

	ix = range(len(Xs))
	random.shuffle(ix)
	split_ix = int(train_size*len(Xs))

	X_train, y_train = [], []
	for i in ix[:split_ix]:
		X_train += Xs[i]
		y_train += ys[i]

	X_val, y_val = [], []
	for i in ix[split_ix:]:
		X_val += Xs[i]
		y_val += ys[i]


	X_train = np.vstack(X_train)
	y_train = np.vstack(y_train)
	X_val = np.vstack(X_val)
	y_val = np.vstack(y_val)

	return X_train, X_val, y_train, y_val
