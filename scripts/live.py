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
from convcube import dbschema
from ModalDB import ModalClient, Video, Frame

if __name__ == '__main__':

	client = ModalClient('./data/db', schema=dbschema)

	#=====[ Step 2: turn on the webcam	]=====
	cap = cv2.VideoCapture(0)
	cv2.namedWindow('DISPLAY')



	loc_convnet = EuclideanConvNet()
	loc_convnet.load('./models/localization/5layer.filter_3333.nfilters_16321632.acc_0035.convnet')
	cube = ConvNetCube(loc_convnet=loc_convnet)

	# video = client.get(Video, 'lanthrop_2')
	# for frame in video.iter_children(Frame):
	# 	image = frame['image']
	# 	output = cube.draw_output(image)
	# 	cv2.imshow('DISPLAY', output)
	# 	key = cv2.waitKey(1)
	# 	while key & 0xFF != ord('q'):
	# 		key = cv2.waitKey(30)



	#=====[ Step 3: capture frames until ESC	]=====
	while True:

		#####[ GRAB IMAGE FROM WEBCAM	]#####
		ret, image = cap.read()
		image = imresize(image, (360, 640))
		output = cube.draw_output(image)

		cv2.imshow('DISPLAY', output)

		key = cv2.waitKey(1)
		# while key & 0xFF != ord('q'):
			# key = cv2.waitKey(30)

		# while cv2.waitKey(1) & 0xFF != ord('q'):
			# pass

		####[ BREAK ON ESCAPE	]#####
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#=====[ Step 4: cleanup	]=====
	cap.release()
	cv2.destroyAllWindows()



