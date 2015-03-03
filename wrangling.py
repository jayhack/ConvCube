import pickle
from ModalDB import ModalClient, Video, Frame
from cs231n.data_utils import load_tiny_imagenet


def put_channels_first(X):
	"""returns X with depth dimension presented first"""
	return X.transpose(0, 3, 1, 2).copy()


def load_transfer_dataset(client, X):
	"""
	TODO: should this iterate? downsample?

	loads dataset for transfer learning: 
		images -> interior corner maps 

	client: ModalDB client
	"""

	for video in client.iter(Video):
		for frame in video.iter_children(Frame):
			image = frame['image']


