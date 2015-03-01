"""
Script: show_data_stats.py
==========================

Description:
------------
	
	prints out stats on labels available from videos


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

from schema import convcube_schema

if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=convcube_schema)

	#=====[ Step 3: iterate through frames	]=====
	ints, exts = [], []
	for video in client.iter(Video):
		print 'Video: %s' % video._id
		total = 0
		for frame in video.iter_children(Frame):
			ext = frame['exterior_points']
			total += len(ext)
		print '	%d exterior points' % total




