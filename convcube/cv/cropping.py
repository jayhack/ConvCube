import numpy as np
import cv2

def crop_image(image, box):
	"""(image, box) -> cropped image"""
	tl, br = box[0], box[1]
	return image[tl[1]:br[1], tl[0]:br[0]]


def point_in_box(pt, box):
	"""returns true if kpt falls in box"""
	x, y = pt
	x1, x2 = box[0][0], box[1][0]
	y1, y2 = box[0][1], box[1][0]
	return (x>=x1) and (x<=x2) and (y>=y1) and (y<=y2)


def crop_points(pts, box):
	"""(pts, box) -> cropped points (new coordinates)"""
	tl, br = box[0], box[1]
	new_pts = []
	for p in pts:
		if point_in_box(p, box):
			new_pts.append((p[0] - tl[0], p[1] - tl[1]))
	return new_pts