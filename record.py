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
import cv2
from ModalDB import ModalClient

from schema import convcube_schema

if __name__ == '__main__':

	#=====[ Step 1: boot up	and create convcube schema ]=====
	modal_client = ModalClient('./data/db', schema=convcube_schema)

