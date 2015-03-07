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
import pickle
import numpy as np
import scipy as sp
import cv2

from ModalDB import Video, Frame

dbschema = {

	Frame: {

		'image':	{
						'mode':'disk',
						'filename':'image.jpg',
						'load_func':lambda p: cv2.imread(p),
						'save_func':lambda x, p: cv2.imwrite(p, x)
					},
		'preprocessed':{
						'mode':'disk',
						'filename':'preprocessed.jpg',
						'load_func':lambda p: cv2.imread(p),
						'save_func':lambda x, p:cv2.imwrite(p, x)
						},
		'interior_points': {
						'mode':'disk',
						'filename':'interior_points.pkl',
						'load_func': lambda p: pickle.load(open(p)),
						'save_func': lambda x, p: pickle.dump(x, open(p, 'w'))
						},
		'exterior_points':{
						'mode':'disk',
						'filename':'exterior_points.pkl',
						'load_func': lambda p: pickle.load(open(p)),
						'save_func': lambda x, p: pickle.dump(x,open(p, 'w'))
						},
		'bounding_box':{
						'mode':'disk',
						'filename':'bounding_box.pkl',
						'load_func': lambda p: pickle.load(open(p)),
						'save_func': lambda x, p: pickle.dump(x,open(p, 'w'))
						},
		'center_points':{
						'mode':'disk',
						'filename':'center_points.pkl',
						'load_func': lambda p: pickle.load(open(p)),
						'save_func': lambda x, p: pickle.dump(x,open(p, 'w'))
						}
	},

	Video: {
		'contains':[Frame]
	}

}