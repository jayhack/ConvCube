from ModalDB import ModalClient
from ModalDB import Video
from ModalDB import Frame


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




