import pickle
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


def load_transfer_dataset(client):
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


	#=====[ Step 3: to numpy arrays	]=====
	X = np.array(X)
	X = put_channels_first(X)
	y = np.array(y)
	return X, y

