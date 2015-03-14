import os
import sys
from scipy.misc import imsave
from convcube.convnets.localization import load_dataset_localization
from ModalDB import ModalClient, Frame, Video

if __name__ == '__main__':

	print '-----> Step 1: loading data'
	X_train, X_val, y_train, y_val = load_dataset_localization()

	print '-----> Step 2: saving train data'
	train_data_dir = './data/caffe_data/localization/train'
	for i in range(X_train.shape[0]):
		image = X_train.transpose(1, 2, 0).copy()
		filename = os.path.join(train_data_dir, '%d.jpg' % i)
		imsave(filename, image)

	print '-----> Step 3: saving train labels'
	


