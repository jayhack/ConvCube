"""
Script: record.py
=================

Description:
------------
	
	Records a new video and stores it

Usage:
------

	python record.py

##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
import time
import pickle
import numpy as np
from scipy.misc import imresize
import cv2
from convcube import ConvNetCube, EuclideanConvNet

if __name__ == '__main__':

	#=====[ Step 2: turn on the webcam	]=====
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('DISPLAY')


	loc_convnet = EuclideanConvNet()
	loc_convnet.load('./models/localization/10s.loc_convnet')
	print loc_convnet
	cube = ConvNetCube(loc_convnet=loc_convnet)

	#=====[ Step 3: capture frames until ESC	]=====
	while True:

		#####[ GRAB IMAGE FROM WEBCAM	]#####
		ret, image = cap.read()
		image = imresize(image, (360, 640))
		print cube.localize(image)

		cv2.imshow('DISPLAY', image)

		#####[ BREAK ON ESCAPE	]#####
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#=====[ Step 4: cleanup	]=====
	cap.release()
	cv2.destroyAllWindows()



