"""
Script: label_bouding_boxes.py
==============================

Description:
------------
	
	Lets you fill in bounding boxes


Usage:
------

	python label_bouding_boxes.py 

##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
import time
import numpy as np
import cv2
from ModalDB import ModalClient, Video, Frame
from convcube.db import dbschema
from convcube.cv.keypoints import normalize_points
from convcube.cv.keypoints import denormalize_points


class Labeler(object):
	"""stateful labeling"""

	def __init__(self):
		self.window_name = 'DISPLAY'
		cv2.namedWindow(self.window_name)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)
		self.drawing = False



	def mouse_callback(self, event, x, y, flags, param):
		"""handles moust input"""
		coord = normalize_points([(x, y)], self.image)[0]

		if not self.drawing:
			if event == cv2.EVENT_LBUTTONDOWN:
				self.box = [coord, coord]
				self.drawing = True

		else:
			self.box[1] = coord
			
			if event == cv2.EVENT_LBUTTONDOWN:
				self.drawing = False



	def draw_box(self, image, box):
		"""draws the box"""
		if not self.box is None:
			tl = denormalize_points([(box[0][0], box[0][1])], self.image)[0]
			br = denormalize_points([(box[1][0], box[1][1])], self.image)[0]
			disp_image = image.copy()
			cv2.rectangle(disp_image, tl, br, (0, 255, 0), thickness=3)
			return disp_image
		else:
			return image
			

	def label(self, image):
		"""labels the image, returning box"""
		self.image = image
		self.box = None
		self.drawing = False

		while True:

			disp_image = self.draw_box(self.image, self.box)
			cv2.imshow(self.window_name, disp_image)

			#=====[ Step 5: wait for ESC to continue	]=====
			key = cv2.waitKey(20)
			if key & 0xFF == 27:
				break

		return self.box






if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=dbschema)

	#=====[ Step 2: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)


	for frame in video.iter_children(Frame):

		#=====[ Get and store data	]=====
		image = frame['image'].copy()
		box = frame['bounding_box']
		print box
		disp_image = image.copy()
		if not box is None and len(box) == 2:
			tl = denormalize_points([(box[0][0], box[0][1])], image)[0]
			br = denormalize_points([(box[1][0], box[1][1])], image)[0]
			cv2.rectangle(disp_image, tl, br, (0, 255, 0), thickness=3)
		cv2.imshow('DISPLAY', disp_image)

		key = cv2.waitKey(10)
		while key != ord('a') and key != 107:
			key = cv2.waitKey(10)
		if key == ord('a'):
			print 'incorrect'
			frame['bounding_box'] = None
		elif key == 107:
			print 'CORRECT'
			frame['bounding_box'] = box



