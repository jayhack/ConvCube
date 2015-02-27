"""
Script: mark_outer_corners.py
=============================

Description:
------------
	
	Given a particular video, will allow you to iterate through and mark 
	corners

Usage:
------

	python mark_outer_corners.py 

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
from cvlabel import *

from schema import convcube_schema

if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=convcube_schema)

	#=====[ Step 2: boot up a labeler	]=====
	def find_interesting_points(image):
		"""finds interesting points to track"""
		feature_params = dict(	maxCorners = 500,
								qualityLevel = 0.1,
								minDistance = 7,
								blockSize = 7 )
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)
		p0 = p0[:,0,:]
		return [tuple(p) for p in list(p0)]


	def get_random_points(image, n=4):
		"""returns a list of n random (x,y) coordinates"""
		xtop, ytop, nchannels = image.shape
		xs = np.random.randint(0, xtop, (n,1))
		ys = np.random.randint(0, ytop, (n,1))
		return [(int(xs[i]), int(ys[i])) for i in range(n)]

	labeler = CVLabeler(find_interesting_points, draw_circle, euclidean_distance, boolean_flip_label)


	#=====[ Step 3: Get video	]=====
	# video_name = raw_input('Name of video to mark: ')
	video_name = 'jay_test_1'
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)


	#=====[ Step 3: iterate through frames	]=====
	cv2.namedWindow('DISPLAY')
	for frame in video.iter_children(Frame):
		image = frame['image'].copy()
		print image.shape
		height, width, nchannels = image.shape
		image = cv2.resize(image, (480, 640), interpolation = cv2.INTER_CUBIC)
		objs, labels = labeler.label(image)





