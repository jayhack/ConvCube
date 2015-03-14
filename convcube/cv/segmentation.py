import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries

def get_segs(image):
    """image -> segs"""
    return slic(image, n_segments=25, enforce_connectivity=True)


################################################################################
####################[ Simple Manipulations on Segs ]############################
################################################################################

def get_covered_segment(segs, pt):
    """(segs, pt) -> segment point falls into"""
    x, y = int(pt[0]), int(pt[1])
    assert y <= segs.shape[0] and x <= segs.shape[1]
    return segs[y, x]


def get_seg_center(segs, ix):
    """(segs, segment index) -> segment center"""
    ys, xs = np.where(segs == i)
    return int(xs.mean()), int(ys.mean())




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


