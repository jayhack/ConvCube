import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.exposure import histogram

def get_segs(image):
    """image -> segs"""
    return slic(image, n_segments=25, enforce_connectivity=True)


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

    hist_h, bins = np.histogram(h, bins=10, range=(0, 255), density=True)
    hist_s, bins = np.histogram(s, bins=10, range=(0, 255), density=True)
    hist_v, bins = np.histogram(v, bins=10, range=(0, 255), density=True)

    return hist_h, hist_s, hist_v


def get_surrounding_colors(image, pt, size=2):
    """(image, pt) -> (hist_h, hist_s, hist_v) of HSV from region of size*2 around pt"""
    x, y = int(pt[0]), int(pt[1])
    region = image[y-size:y+size, x-size:x+size]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(hsv.shape[0] * hsv.shape[1], hsv.shape[2])
    h, s, v = pix[:, 0], pix[:, 1], pix[:, 2]

    hist_h, bins = np.histogram(h, bins=10, range=(0, 255))
    hist_s, bins = np.histogram(s, bins=10, range=(0, 255), density=True)
    hist_v, bins = np.histogram(v, bins=10, range=(0, 255), density=True)

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
    for i in range(segs.max() + 1):
        mask = segs == i
        masked = image[mask]
        
        #=====[ Pixel Values ]=====
        pix_avg = masked.mean(axis=0)
        pix_std = masked.std(axis=0)
        pix_max = masked.max(axis=0)
        pix_min = masked.min(axis=0)
        pix_range = pix_max - pix_min

        #=====[ Moments ]=====
        Y, X = np.where(mask) 
        ylen = Y.max() - Y.min()
        xlen = X.max() - Y.max()
        yavg, ystd = Y.mean(), Y.std()
        xavg, xstd = X.mean(), X.std()
        area = len(X)
        
        #=====[ Put together ]====
        f = [ylen, xlen, yavg, ystd, xavg, xstd, area]
        f += pix_avg.tolist()
        f += pix_std.tolist()
        f += pix_max.tolist()
        f += pix_min.tolist()
        f += pix_range.tolist()
        features.append(f)
    
    return np.array(features)


def segs2centers(image, segs, clf, threshold=0.3):
    """(segs, classifier) -> list of centers of cube faces"""
    X = featurize_segs(image, segs)
    y = clf.predict_proba(X)[:,1] > threshold
    centers = [get_seg_center(segs, i) for i in np.where(y)[0]]
    return centers


def get_seg_centers(image, clf, threshold=0.3):
    """(image, classifier) -> list of seg centers"""
    segs = get_segs(image)
    centers = segs2centers(image, segs, clf, threshold=threshold)
    features = featurize_segs(image, segs)
    labels = (clf.predict_proba(features)[:,1] >= threshold)
    return [c for c, l in zip(centers, labels) if l]


