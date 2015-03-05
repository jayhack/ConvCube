class Cube(object):
	"""
	Class: Cube
	-----------
	Abstract class for CV on rubiks cube.
	"""
	def __init__(self):
		pass

	def load(self, path):
		"""loads classifiers"""
		raise NotImplementedError

	def save(self, path):
		"""saves consumed data"""
		raise NotImplementedError

	def update(self, frame):
		"""updates current state"""
		raise NotImplementedError

	def draw(self, frame):
		"""draws current state on frame"""
		raise NotImplementedError

