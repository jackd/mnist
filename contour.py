import numpy as np


def image_to_contours(image, level=0.5):
    from skimage import measure
    return measure.find_contours(image, level=level)


def sample_contours(contours, n_points):
    lengths = []
    offsets = []
    normals = []
    points = []
    for contour in contours:
        contour = contour.astype(np.float32)
        offset = contour[1:] - contour[:-1]
        length = np.sqrt(np.sum(offset**2, axis=-1, keepdims=True))
        normals.append(
            np.stack((-offset[:, 1], offset[:, 0]), axis=-1) / length)
        offsets.append(offset)
        lengths.append(length)
        points.append(contour[:-1])
    lengths = np.concatenate(lengths, axis=0)
    lengths = np.squeeze(lengths, axis=-1)
    offsets = np.concatenate(offsets, axis=0)
    points = np.concatenate(points, axis=0)
    normals = np.concatenate(normals, axis=0)
    lengths /= np.sum(lengths)
    counts = np.random.multinomial(n_points, lengths)
    indices = np.concatenate(
        tuple((i,)*c for i, c in enumerate(counts)),
        axis=0).astype(np.int32)

    r = np.random.rand(n_points, 1).astype(np.float32)
    p = points[indices] + r*offsets[indices]
    n = normals[indices]

    return p, n
