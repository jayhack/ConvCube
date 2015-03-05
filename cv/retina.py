import numpy as np
import cv2

def FoveatedRetina(object):
	"""
	Class: FoveatedRetina
	---------------------
	Transforms images from flat entities to a 'retinal' representation

	supports 
		retina.get_layers(attention) #gets all simultaneously
		retina.get_layer(0, attention) #gets just one, so you can update attention
		...
		retina.get_layer(1, new_attention)
		...
		etc.
	"""

	def __init__(self, num_layers, layer_sizes, layer_resolutions):
		"""
		- num_layers: # of layers in retinal representation
		- layer_sizes: spans of layers in original image pixels
		- layer_resolutions: resolutions of each layer
		"""
		raise NotImplementedError
		self.num_layers = num_layers
		self.layer_sizes = layer_sizes
		self.centers = [(0,0) for l in layer_sizes]


	def compute_centers(self, attention):
		"""center of focus -> new centers of layer"""
		raise NotImplementedError


	def extract(self, image, centers):
		"""extract image regions from image"""
		raise NotImplementedError


	def transform(self, image):
		"""image -> retinal representation"""
		raise NotImplementedError


	def show(self, retinal_image):
		"""retinal_image -> displayable version"""
		raise NotImplementedError
