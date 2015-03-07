from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
				#=====[ cs231n fast layers	]=====
				Extension(	
							'convcube/cs231n/im2col_cython', 
							['convcube/cs231n/im2col_cython.pyx'], 
							include_dirs=[np.get_include()]
						),
			]

setup(
		name="convcube",
		version="0.2",
		author="Jay Hack",
		author_email="jhack@stanford.edu",
		description="Computer vision + convnets for Rubiks Cube AR",
		ext_modules=cythonize(extensions)
)
