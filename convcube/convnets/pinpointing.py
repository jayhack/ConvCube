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
from ModalDB import Frame, Video

def pinpointing_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (46, 80):
		image = resize(image, (46, 80))
	return image


def get_X_pinpointing(image, box):
	"""image -> array for input to convnet"""
	box = denormalize_points(box, image)
	if (box[1][0] - box[0][0] > 5) and (box[1][1] - box[0][1] > 5):
		image = crop_image(image, box)
		return put_channels_first(pinpointing_resize(image))
	else:
		return None


def get_y_pinpointing(center_points):
	"""image, kpts -> coordinates of center. -1s when it doesn't exist"""
	default = -np.ones((4,2))
	wy, bg, ro = default, default, default

	if 'wy' in center_points:
		wy = tuples2array(center_points['wy'])
	if 'bg' in center_points:
		bg = tuples2array(center_points['bg'])
	if 'or' in center_points:
		ro = tuples2array(center_points['or'])

	kpts = np.hstack([wy, bg, ro])
	kpts = np.mean(kpts, axis=0)
	return kpts



def get_convnet_inputs_pinpointing(frame):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_pinpointing(frame['image'], frame['bounding_box'])
	y = get_y_pinpointing(frame['center_points'])
	return X, y


def load_dataset_pinpointing(client, train_size=0.9):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []

	for frame in iter_labeled_frames(client):

		center_points = frame['center_points']
		if type(center_points) == dict and len(center_points.keys()) > 0:

			X_, y_= get_convnet_inputs_pinpointing(frame)
			if not X_ is None and not y_ is None:
				X.append(X_)
				y.append(y_)

	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)
