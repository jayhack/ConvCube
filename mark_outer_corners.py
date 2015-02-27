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
		feature_params = dict(	maxCorners = 100,
								qualityLevel = 0.3,
								minDistance = 7,
								blockSize = 7 )
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)
		return [tuple(p) for p in list(p0)]
	labeler = CVLabeler(find_interesting_points, draw_circle, euclidean_distance, boolean_flip_label)


	#=====[ Step 3: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)


	#=====[ Step 3: iterate through frames	]=====
	cv2.namedWindow('DISPLAY')
	for frame in video.iter_children(Frame):

		cv2.imshow('DISPLAY', frame['image'])

		#####[ BREAK ON ESCAPE	]#####s
		if cv2.waitKey(20) & 0xFF == ord('q'):
			break



