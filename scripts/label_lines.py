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
from ModalDB import ModalClient, Video, Frame
from convcube import dbschema
from convcube.cv.keypoints import denormalize_points
from convcube.cv.keypoints import normalize_points
from convcube.cv.cropping import crop_image
import matplotlib.pyplot as plt


class Labeler(object):
	"""stateful labeling"""

	def __init__(self):
		self.window_name = 'DISPLAY'
		# self.fig = plt.figure()
		# self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		cv2.namedWindow(self.window_name)
		cv2.setMouseCallback(self.window_name, self.mouse_callback)

	# def onclick(self, event):
	#	"""matplotlib callback
	# 	print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata)
	# 	self.points.append((int(event.xdata), int(event.ydata)))


	def mouse_callback(self, event, x, y, flags, param):
		"""handles moust input"""

		if event == cv2.EVENT_LBUTTONDOWN:
			self.points.insert(0, (x,y))

		if len(self.points) > 4:
			self.points.pop()


	def draw_points(self, image, box):
		"""draws the box"""
		for point in self.points:
			cv2.circle(image, point, 5, (0, 255, 0), thickness=2)
			

	def label(self, image):
		"""labels the image, returning box"""
		self.image = image
		self.points = []

		while True:

			disp_img = self.image.copy()
			self.draw_points(disp_img, self.points)
			cv2.imshow(self.window_name, disp_img)

			#=====[ Step 5: wait for ESC to continue	]=====
			key = cv2.waitKey(20)
			if key & 0xFF == 27:
				break

		if len(self.points) == 4:
			return normalize_points(self.points, image)
		else:
			return None



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
			box = frame['bounding_box']
			box = denormalize_points(box, frame['image'])
			image = crop_image(frame['image'].copy(), box)
			image = imresize(image, 300)
			points = labeler.label(image)
			if not points is None:
				frame['center_points'] = points


