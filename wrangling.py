import pickle
import random
import numpy as np
from ModalDB import ModalClient, Video, Frame
from preprocess import imagenet_resize
from ML import make_output_heatmap, make_output_coords


def put_channels_first(X):
	"""returns X with depth dimension presented first"""
	return X.transpose(0, 3, 1, 2).copy()


def has_label(frame):
	"""ModalDB.Frame -> has label or not"""
	if not frame['interior_points'] is None:
		return len(frame['interior_points']) > 0


def iter_labeled_frames(client):
	"""iterator over frames that have labels"""
	for frame in client.iter(Frame):
		if has_label(frame):
			yield frame
			

def load_transfer_dataset(client, train=0.75):
	"""
	TODO: should this iterate? downsample?

	loads dataset for transfer learning: 
		images -> interior corner maps 

	client: ModalDB client
	"""
	X, y = [], []

	for video in client.iter(Video):
		for frame in video.iter_children(Frame):
			if has_label(frame):
		
				#=====[ Step 1: format image	]=====
				image = frame['image']
				image = imagenet_resize(image)
				X.append(image)

				#=====[ Step 2: get label	]=====
				kpts = frame['interior_points']
				output = make_output_coords(kpts)
				y.append(output)

	#=====[ Step 3: to numpy arrays ]=====
	X = put_channels_first(np.array(X))
	y = np.array(y)

	#=====[ Step 4: shuffle	]=====
	shuffle_ix = np.random.permutation(range(X.shape[0]))
	X = X[shuffle_ix,:,:,:].copy()
	y = y[shuffle_ix,:].copy()

	#=====[ Step 5: test and validation	]=====
	n = X.shape[0]
	split_ix = int(n*train)
	X_train = X[:split_ix,:,:,:]
	y_train = y[:split_ix,]
	X_val = X[split_ix:,:,:,:]
	y_val = y[split_ix:,:]

	return X_train, y_train, X_val, y_val

