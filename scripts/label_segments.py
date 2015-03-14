"""
Script: label_lines.py
======================

Description:
------------
	
	lets you plug in points within bounding boxes


Usage:
------

	python label_lines.py

##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
import time
import numpy as np
from scipy.misc import imresize
import cv2
from skimage.segmentation import mark_boundaries
from ModalDB import ModalClient, Video, Frame
from convcube import dbschema
from convcube.convnets.pinpointing import get_X_pinpointing
from convcube.cv.preprocess import resize
from convcube.cv.keypoints import denormalize_points
from convcube.cv.keypoints import normalize_points
from convcube.cv.cropping import crop_image
from convcube.cv.segmentation import get_kmeans_segs
from convcube.cv.segmentation import featurize_segs


class Labeler(object):
	"""stateful labeling"""

	def __init__(self):
		self.window_name = 'DISPLAY'
		cv2.namedWindow(self.window_name)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)


	def mouse_callback(self, event, x, y, flags, param):
		"""handles moust input"""
		coords = normalize_points([(x, y)], self.disp_img)
		x, y = denormalize_points(coords, self.image)[0]

		if event == cv2.EVENT_LBUTTONDOWN:
			seg_ix = self.segs[y,x]
			if seg_ix in self.pos:
				self.pos.remove(seg_ix)
			else:
				self.pos.add(seg_ix)


	def draw_segs(self):
		"""draws the box"""
		bounds_img = mark_boundaries(self.image, self.segs)
		for p in self.pos:
			mask = self.segs == p
			bounds_img[mask] = (0, 255, 0)
		return cv2.resize(bounds_img, (400,400))
			

	def label(self, image):
		"""labels the image, returning box"""
		#=====[ Step 1: get segments	]=====
		self.image = image.astype(np.uint8)
		self.segs = get_kmeans_segs(image, n_clusters=8)


		self.pos, self.neg = set([]), set([])
		print self.segs.min(), self.segs.max()

		#=====[ Step 2: label segments	]=====
		while True:

			self.disp_img = self.draw_segs()
			cv2.imshow(self.window_name, self.disp_img)

			key = cv2.waitKey(20)
			if key & 0xFF == 27:
				break

		if len(self.pos) == 0:
			return None

		self.neg = set(range(self.segs.max() + 1)).difference(self.pos)


		#=====[ Step 3: save segments	]=====
		segments = {}
		segments['segs'] = self.segs
		segments['pos'] = self.pos
		segments['neg'] = self.neg
		return segments




if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=dbschema)

	#=====[ Step 2: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)

	#=====[ Step 3: setup display	]=====
	labeler = Labeler()
	for frame in video.iter_children(Frame):
		if not frame['bounding_box'] is None:

			#=====[ Get and store data	]=====
			X = get_X_pinpointing(frame['image'], frame['bounding_box'])[0,:,:,:]
			if X.max() <= 1.0:
				X *= (255.0/2.0)
				X += 255.0/2.0

			image = X.transpose(1, 2, 0)
			segments = labeler.label(image)
			if not segments is None:
				frame['segments'] = segments



