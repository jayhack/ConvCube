import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.exposure import histogram
from sklearn.cross_validation import train_test_split
from ModalDB import ModalClient, Frame



def get_kmeans_image(image, n_clusters=10):
    """image -> kmeans image"""
    pix = image.reshape((image.shape[0]*image.shape[1], image.shape[2])).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10)
    labels = km.fit_predict(pix)
    labels_img = labels.reshape((image.shape[0], image.shape[1]))
    km_img = np.zeros_like(image)
    for i in range(n_clusters):
        km_img[labels_img == i] = km.cluster_centers_[i,:]
    return km_img


def get_segs(image, n_segments=25):
    """image -> segs"""
    return slic(image, n_segments=n_segments, enforce_connectivity=True, compactness=30)


def get_kmeans_segs(image, n_clusters=10, n_segments=25):
    """image -> segs from kmeans image"""
    km_img = get_kmeans_image(image, n_clusters=n_clusters)
    segs = get_segs(image, n_segments=n_segments)
    return segs




################################################################################
####################[ Simple Manipulations on Segs ]############################
################################################################################



def get_covered_seg(segs, pt):
    """(segs, pt) -> segment point falls into"""
    x, y = int(pt[0]), int(pt[1])
    assert y <= segs.shape[0] and x <= segs.shape[1]
    return segs[y, x]


def get_seg_center(segs, ix):
    """(segs, segment index) -> segment center"""
    ys, xs = np.where(segs == ix)
    return int(xs.mean()), int(ys.mean())


def get_seg_colors(image, segs, ix):
    """(image, segs, segment index) -> hist over h from HSV of segment"""
    mask = (segs == ix)
    pix = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[mask]
    h, s, v = pix[:, 0], pix[:, 1], pix[:, 2]

    hist_h, bins = np.histogram(h, bins=10, range=(0, 255))
    hist_s, bins = np.histogram(s, bins=10, range=(0, 255))
    hist_v, bins = np.histogram(v, bins=10, range=(0, 255))

    return hist_h, hist_s, hist_v


def get_surrounding_colors(image, pt, size=2):
    """(image, pt) -> (hist_h, hist_s, hist_v) of HSV from region of size*2 around pt"""
    x, y = int(pt[0]), int(pt[1])
    region = image[y-size:y+size, x-size:x+size]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(hsv.shape[0] * hsv.shape[1], hsv.shape[2])
    h, s, v = pix[:, 0], pix[:, 1], pix[:, 2]

    hist_h, bins = np.histogram(h, bins=10, range=(0, 255))
    hist_s, bins = np.histogram(s, bins=10, range=(0, 255))
    hist_v, bins = np.histogram(v, bins=10, range=(0, 255))

    hist_h = hist_h.astype(np.float32) / hist_h.sum().astype(np.float32)
    hist_s = hist_s.astype(np.float32) / hist_s.sum().astype(np.float32)
    hist_v = hist_v.astype(np.float32) / hist_v.sum().astype(np.float32)

    return hist_h, hist_s, hist_v


################################################################################
####################[ ML on Segs ]##############################################
################################################################################

def featurize_segs(image, segs):
    """(image, segs) -> list of features for each segment"""
    features = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(segs.max() + 1):
        
        #=====[ Pixel Hist ]=====
        h_hist, s_hist, v_hist = get_seg_colors(image, segs, i)

        #=====[ Moments ]=====
        mask = (segs == i).astype(np.uint8)
        moments = cv2.moments(mask, binaryImage=True)
        moments = moments.values()
        
        #=====[ Put together ]====
        f = []
        f += moments
        f += h_hist.tolist()
        f += s_hist.tolist()
        f += v_hist.tolist()
        features.append(f)
    
    return np.array(features)


def segs2centers(image, segs, clf, threshold=0.3):
    """(segs, classifier) -> list of centers of cube faces"""
    X = featurize_segs(image, segs)
    y = clf.predict_proba(X)[:,1] > threshold
    centers = [get_seg_center(segs, i) for i in np.where(y)[0]]
    return centers


def get_seg_centers(image, clf, threshold=0.3, topn=None):
    """(image, classifier) -> list of seg centers"""
    segs = get_kmeans_segs(image)
    X = featurize_segs(image, segs)
    y_pred = clf.predict_proba(X)[:,1]

    if topn is None:
        centers = [get_seg_center(segs, i) for i in np.where(y_pred)[0]]

    else:
        sorted_ix = np.argsort(y_pred)[::-1]
        centers = [get_seg_center(segs, sorted_ix[i]) for i in range(topn)]

    return centers




################################################################################
####################[ ML on Segs ]##############################################
################################################################################

def load_dataset_segmentation(client, train_size=0.85):
    """returns X_train, X_val, y_train, y_val for segmentation"""
    X, y = [], []

    for frame in client.iter(Frame):

        if not frame['segments'] is None and type(frame['segments']) == dict:

            image = frame['image']
            segs = frame['segments']['segs']
            pos = np.array(list(frame['segments']['pos'])).astype(np.uint8)
            neg = np.array(list(frame['segments']['neg'])).astype(np.uint8)

            X_ = featurize_segs(image, segs)
            y_ = np.zeros((X_.shape[0],))
            y_[pos] = 1

            X.append(X_)
            y.append(y_)

    X = np.vstack(X)
    y = np.concatenate(y)
    return train_test_split(X, y, train_size=train_size)

