"""
Script: label_data.py
=====================

Description:
------------
	
	Iterates through a video in order for you to mark corners


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
from cvlabel import CVLabeler, euclidean_distance, draw_func, label_func

from schema import convcube_schema
from preprocess import resize, grayscale, resize_grayscale
from keypoints import interesting_points


@label_func(valid_types=[type(None), str])
def mark_interior_exterior(event, label):
	"""None -> 'interior' -> 'exterior' -> None"""
	if label is None:
		return 'interior'
	elif label is 'interior':
		return 'exterior'
	elif label is 'exterior':
		return None
	else:
		raise TypeError("Not recognized label: %s" % str(label))


@draw_func(valid_types=tuple, color_map={ 	'interior':(255, 0, 0), #blue
											'exterior':(0, 255, 0), #green
											None:(0, 0, 255)  		#red
										})
def draw_rubiks_points(disp_image, obj, color, radius=3, thickness=1):
	"""three colors for interior, exterior, nothing"""
	cv2.circle(disp_image, obj, radius, color=color, thickness=thickness)



if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=convcube_schema)

	#=====[ Step 2: boot up a labeler	]=====
	labeler = CVLabeler(	interesting_points, 
							draw_rubiks_points, 
							euclidean_distance, 
							mark_interior_exterior)


	#=====[ Step 3: Get video	]=====
	video_name = raw_input('Name of video to mark: ')
	print 'Video name: %s' % video_name
	video = client.get(Video, video_name)

	#=====[ Step 3: iterate through frames	]=====
	cv2.namedWindow('DISPLAY')
	cv2.resizeWindow('DISPLAY', 2560, 1440)
	ints, exts = [], []
	for frame in video.iter_children(Frame):

		#=====[ Get and store data	]=====
		image = frame['image'].copy()
		points, labels = labeler.label(image)
		interior_points = [p for p,l in zip(points,labels) if l == 'interior']
		exterior_points = [p for p,l in zip(points,labels) if l == 'exterior']
		frame['interior_points'] = interior_points
		frame['exterior_points'] = exterior_points



