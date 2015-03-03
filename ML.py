import numpy as np
import cv2

def construct_kpts_label_image(image, kpts):
    """(image, keypoints) -> image containing corners"""
    kpts = np.array(kpts).astype(np.uint16)
    label_image = np.zeros((image.shape[0], image.shape[1]))
    label_image[kpts[:,1], kpts[:,0]] = 1
    label_image = cv2.GaussianBlur(label_image, (5,5), 2)
    return label_image


def construct_transfer_dataset(iter_image_kpts):
	"""
		(frames, keypoints) iterable -> X, y training data and labels for ML

		data should be in form 
	"""
	raise NotImplementedError
