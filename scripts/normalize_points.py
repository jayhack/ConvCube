"""
Script: normalize_points.py
===========================

Description:
------------
	
	convert points to normalized in db


##################
Jay Hack
Winter 2015
jhack@stanford.edu
##################
"""
from ModalDB import ModalClient, Video, Frame
from convcube import dbschema
from convcube.cv.keypoints import normalize_points




if __name__ == '__main__':

	#=====[ Step 1: boot up	ModalClient ]=====
	client = ModalClient('./data/db', schema=dbschema)
	for frame in client.iter(Frame):

		pts = frame['interior_points']
		if not pts is None:
			if len(pts) > 0:
				frame['interior_points'] = normalize_points(pts, frame['image'])

		pts = frame['exterior_points']
		if not pts is None:
			if len(pts) > 0:
				frame['exterior_points'] = normalize_points(pts, frame['image'])
