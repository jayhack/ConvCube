from setuptools import setup, find_packages

setup(
		name="convcube",
		version="0.1",
		author="Jay Hack",
		author_email="jhack@stanford.edu",
		description="Computer vision + convnets for Rubiks Cube AR",
		packages=find_packages(),
		include_package_data=True,
		install_requires=[
							'numpy',
							'scipy',
							'scikit-learn'
							# 'cv2'
						]
)
