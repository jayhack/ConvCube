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
from convcube import dbschema


class Labeler(object):
	"""stateful labeling"""

	def __init__(self):
		self.window_name = 'DISPLAY'
		cv2.namedWindow(self.window_name)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)


	def normalize_coord(self, x, y):
		"""image coordinates -> normalized (0 to 1, floats)"""
		x, y = float(x), float(y)
		H, W = float(self.image.shape[0]), float(self.image.shape[1])
		return (x / W, y / H)


	def unnormalize_coord(self, x, y):
		"""normalized coords -> image coords"""
		H, W = float(self.image.shape[0]), float(self.image.shape[1])
		return int(x*W), int(y*H)


	def mouse_callback(self, event, x, y, flags, param):
		"""handles moust input"""
		coord = self.normalize_coord(x, y)

		if event == cv2.EVENT_LBUTTONDOWN:
			if self.box is None:
				self.box = [coord, coord]
			else:
				self.box[0] = coord

		elif event == cv2.EVENT_LBUTTONUP:
			if not self.box is None:
				self.box[1] = coord


	def draw_box(self, image, box):
		"""draws the box"""
		if not self.box is None:
			tl = self.unnormalize_coord(box[0][0], box[0][1])
			br = self.unnormalize_coord(box[1][0], box[1][1])
			disp_image = image.copy()
			cv2.rectangle(disp_image, tl, br, (0, 255, 0), thickness=3)
			return disp_image
		else:
			return image
			

	def label(self, image):
		"""labels the image, returning box"""
		self.image = image
		self.box = None

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
	client = ModalClient('./data/db', schema=schema)
	print convcube_schema

	#=====[ Step 2: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)

	#=====[ Step 3: setup display	]=====
	labeler = Labeler()


	for frame in video.iter_children(Frame):

		print client.schema.schema_dict
		#=====[ Get and store data	]=====
		image = frame['image'].copy()
		box = labeler.label(image)
		print box
		frame['bounding_box'] = box



