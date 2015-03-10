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
from convcube.cv.keypoints import denormalize_points
from convcube.cv.keypoints import normalize_points
from convcube.cv.cropping import crop_image
from convcube.cv.segmentation import get_segments
from convcube.cv.segmentation import featurize_segments


class Labeler(object):
	"""stateful labeling"""

	def __init__(self):
		self.window_name = 'DISPLAY'
		cv2.namedWindow(self.window_name)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)


	def mouse_callback(self, event, x, y, flags, param):
		"""handles moust input"""

		if event == cv2.EVENT_LBUTTONDOWN:
			point = x, y
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
		return bounds_img
			

	def label(self, image):
		"""labels the image, returning box"""
		self.image = image
		self.segs = get_segments(image)
		self.pos, self.neg = set([]), set([])

		while True:

			disp_img = self.draw_segs()
			cv2.imshow(self.window_name, disp_img)

			#=====[ Step 5: wait for ESC to continue	]=====
			key = cv2.waitKey(20)
			if key & 0xFF == 27:
				break



if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=dbschema)

	#=====[ Step 2: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)

	#=====[ Step 3: Get color	]=====
	color_name = raw_input('Color to mark: (wy, bg, or)')
	assert color_name in ['wy', 'bg', 'or']
	print 'Color name: %s' % color_name


	#=====[ Step 3: setup display	]=====
	labeler = Labeler()
	for frame in video.iter_children(Frame):
		if not frame['bounding_box'] is None:

			#=====[ Get and store data	]=====
			X = get_X_pinpointing(frame)
			if X.max() <= 1.0:
				X *= (255.0/2.0)
				X += 255.0/2.0
			image = X.transpose(1, 2, 0)
			pos, neg = labeler.label(image)


