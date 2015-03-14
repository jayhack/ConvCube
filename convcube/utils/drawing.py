import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_images(images, n=4):
    """displays n images in a row via matplotlib
    """
    plt.rcParams['figure.figsize'] = 20, 50
    fig, axes = plt.subplots(nrows=1, ncols=min(n, 4))
    for i, img in enumerate(images[:n]):
        axes[i].imshow(img, cmap=plt.cm.gray)
        axes[i].set_xticks([])
        axes[i].set_yticks([])


def draw_points(frame, points, labels=None):
	"""
		Description:
		------------
		(frame, points, labels) -> image with points drawn 

		Args:
		-----
		- frame: frame to draw on 
		- points: tuple or list of tuples
		- labels: list of labels.
					{0, None, False} 	=> red, faint
					{1, True} 			=> green, bold
					{2} 				=> blue, faint
	"""
	#=====[ Step 1: standardize points	]=====
	if type(points) == tuple:
		points = [points]

	#=====[ Step 2: standardize labels 	]=====
	if labels is None:
		labels = [0 for p in points]
	elif type(labels) == list:
		for i in range(len(labels)):
			if (labels[i] is None) or (labels[i] == False):
				labels[i] = 0
			elif (labels[i] == True):
				labels[i] = 1
	if not all([type(l) == int for l in labels]):
		raise TypeError("Didn't understand labels")

	if not len(points) == len(labels):
		raise TypeError("Must be same number of points and labels or no labels!")

	#=====[ Step 3: get point properties	]=====
	red, green, blue = (0, 0, 255), (0, 255, 0), (255, 0, 0)
	lmap = {
				0:{'color':red, 'thickness':1, 'radius':3},
				1:{'color':green, 'thickness':2, 'radius':5},
				2:{'color':blue, 'thickness':1, 'radius':3}
	}

	#=====[ Step 4: draw circles around points	]=====
	disp_img = frame.copy()
	for p, l in zip(points, labels):
		cv2.circle(disp_img, (int(p[0]), int(p[1])), lmap[l]['radius'], color=lmap[l]['color'], thickness=lmap[l]['thickness'])

	return disp_img
