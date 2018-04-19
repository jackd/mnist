import numpy as np
from scipy.ndimage.filters import convolve


def image_to_cloud(image):
    return np.where(image)


def sample_image(image, n_points):
    y, x = image_to_cloud(image)
    n = len(y)
    if n == 0:
        raise ValueError('Cannot sample empty image')
    indices = np.random.choice(range(n), n_points, replace=True)
    return y[indices], x[indices]


_vn = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.bool)
_moore = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.bool)


def boundary_image(image, neighborhood='vn'):
    if image.dtype != np.bool:
        image = image > 0
    empty = np.logical_not(image)
    if neighborhood == 'vn':
        filt = _vn
    elif neighborhood == 'moore':
        filt = _moore
    else:
        raise ValueError('Invalid neighborhood "%s"' % neighborhood)

    boundary = np.logical_and(
        image, convolve(empty, filt, mode='constant', cval=True))
    return boundary
