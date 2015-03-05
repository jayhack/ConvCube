import pickle
import random
import numpy as np
from ModalDB import ModalClient
from ModalDB import Video
from ModalDB import Frame
from preprocess import image2localization_array


################################################################################
####################[ ITERATION UTILS ]#########################################
################################################################################

def has_label(frame):
	"""ModalDB.Frame -> has label or not"""
	if not frame['interior_points'] is None:
		return len(frame['interior_points']) > 0


def iter_labeled_frames(client):
	"""iterator over frames that have labels"""
	for frame in client.iter(Frame):
		if has_label(frame):
			yield frame
			

def iter_unlabeled_frames(client):
	"""iterator over random frames without labels"""
	for frame in client.iter(Frame):
		if not has_label(frame):
			yield frame





################################################################################
####################[ LABEL FACTORY ]###########################################
################################################################################


def get_label_heatmap(image, kpts):
	"""(image, kpts) -> output of map

		TODO: this needs to be a 1d array...
	"""
	raise NotImplementedError
	kpts = np.array(kpts).astype(np.uint16)
	label_image = np.zeros((image.shape[0], image.shape[1]))
	label_image[kpts[:,1], kpts[:,0]] = 1
	label_image = cv2.GaussianBlur(label_image, (5,5), 2)
	return label_image


def get_convnet_inputs(frame):
	"""ModalDB.Frame -> (X, y_localization, y_heatmap)"""
	X = image2convnet_array(frame['image'])
	y_localization = get_label_localization(frame['interior_points'])
	y_heatmap = get_label_heatmap(frame['image'])
	return X, y_localization, y_heatmap




