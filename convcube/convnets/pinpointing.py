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

def get_border(image, ideal_ratio=1):
	"""
		given image, determines how large its border should be to get it to ratio of 46/80
	"""
	H,W,C = float(image.shape[0]), float(image.shape[1]), float(image.shape[2])
	img_ratio = H / W
	buf = 0.05

	#=====[ Case: image is too wide...	]=====
	if img_ratio - ideal_ratio < buf:
		ideal_height = ideal_ratio*W
		top, bottom = int(abs(H - ideal_height)/2), int(abs(H - ideal_height)/2)
		image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, 0)


	#=====[ Case: image is too tall...	]=====
	elif img_ratio - ideal_ratio > buf:
		ideal_width = H / ideal_ratio
		left, right = int(abs(W - ideal_width)/2), int(abs(W - ideal_width)/2)
		image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT, 0)

	return put_channels_first(pinpointing_resize(image))





def pinpointing_resize(image):
	"""image -> (46,80,3) shaped-image (46 for even height). idempotent"""
	if not image.shape[:2] == (64, 64):
		image = resize(image, (64, 64))
	return image



def get_X_pinpointing(image, box):
	"""image -> array for input to convnet"""
	box = denormalize_points(box, image)
	if (box[1][0] - box[0][0] > 5) and (box[1][1] - box[0][1] > 5):

		image = crop_image(image, box).astype(np.float32)
		image = pinpointing_resize(image)

		# image *= (2.0/255.0)
		# image -= 1.0

		return put_channels_first(image)

	else:
		return None


def get_y_pinpointing(center_points, name):
	"""image, kpts -> coordinates of center. -1s when it doesn't exist"""
	assert name in ['all', 'bounding_box', 'center', 'top']

	default = -np.ones((4,2))
	wy, bg, ro = default, default, default

	if 'wy' in center_points:
		wy = tuples2array(center_points['wy'])
	if 'bg' in center_points:
		bg = tuples2array(center_points['bg'])
	if 'or' in center_points:
		ro = tuples2array(center_points['or'])

	#=====[ Regular	]=====
	if name == 'all':
		wy = wy.reshape((1, 8))
		bg = bg.reshape((1, 8))
		ro = ro.reshape((1, 8))
		kpts = np.hstack([wy, bg, ro])
		return kpts

	#=====[ Bounding Box	]=====
	elif name == 'bounding_box':
		wy = np.array([wy.min(axis=0), wy.max(axis=0)]).reshape((1, 4))
		bg = np.array([bg.min(axis=0), bg.max(axis=0)]).reshape((1, 4))
		ro = np.array([ro.min(axis=0), ro.max(axis=0)]).reshape((1, 4))
		kpts = np.hstack([wy, bg, ro])
		return kpts

	elif name == 'center':
		kpts = np.hstack([wy, bg, ro])
		kpts = np.mean(kpts, axis=0).reshape((1, 6))
		return kpts

	elif name == 'top':
		wy = wy[np.argmin(wy[:,0]),:]
		bg = bg[np.argmin(bg[:,0]),:]
		ro = ro[np.argmin(ro[:,0]),:]
		kpts = np.hstack([wy, bg, ro])
		return kpts




def get_convnet_inputs_pinpointing(frame, name):
	"""ModalDB.Frame -> (X, y_localization)"""
	X = get_X_pinpointing(frame['image'], frame['bounding_box'])
	y = get_y_pinpointing(frame['center_points'], name)
	return X, y


def load_dataset_pinpointing(client, name='center', train_size=0.9):
	"""returns dataset for localization: images -> X_train, X_val, y_train, y_val"""
	X, y = [], []

	for frame in iter_labeled_frames(client):

		center_points = frame['center_points']
		if type(center_points) == dict and len(center_points.keys()) > 0:

			X_, y_= get_convnet_inputs_pinpointing(frame, name)
			if not X_ is None and not y_ is None:
				X.append(X_)
				y.append(y_)

	X, y = np.vstack(X), np.vstack(y)
	return train_test_split(X, y, train_size=train_size)
