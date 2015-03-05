"""
Script: make_caffe_data.py
==========================

Description:
------------
	
	Makes a data directory containing images and labels


Usage:
------

	python mark_outer_corners.py 

##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
import os
import time
import numpy as np
import cv2
from ModalDB import ModalClient, Video, Frame
from cvlabel import CVLabeler, euclidean_distance, draw_func, label_func

from schema import convcube_schema
from preprocess import resize, grayscale, resize_grayscale, imagenet_resize
from keypoints import interesting_points
from wrangling import iter_labeled_frames
from ML import make_output_heatmap


if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=convcube_schema)

	data_dir = os.path.join('./data/caffe')
	for frame in iter_labeled_frames(client):
		image = frame['image'].copy()
		preprocessed_img = imagenet_resize(image)
		output_img = make_output_heatmap(image)

		#=====[ Get and store data	]=====
		image = frame['image'].copy()
		points, labels = labeler.label(image)
		interior_points = [p for p,l in zip(points,labels) if l == 'interior']
		exterior_points = [p for p,l in zip(points,labels) if l == 'exterior']
		frame['interior_points'] = interior_points
		frame['exterior_points'] = exterior_points



