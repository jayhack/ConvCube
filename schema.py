"""
Schema.py: define your database schema 
======================================

Description:
------------
	
	Describes how different components are stored 
	and accessed


##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
import cPickle
import numpy as np
import scipy as sp
import cv2

from ModalDB import Video, Frame

convcube_schema = {

	Frame: {

		'image':	{
						'mode':'disk',
						'filename':'image.jpg',
						'load_func':lambda p: cv2.imread(p),
						'save_func':lambda x, p: cv2.imwrite(p, x)
					},
	},

	Video: {
		'contains':[Frame]
	}

}