import os
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


def _save_mode_data(group, mode, n_samples):
    from dataset import get_dataset
    from progress.bar import IncrementalBar
    dataset = get_dataset(mode)
    n = len(dataset)
    with dataset:
        mode_group = group.create_group(mode)
        points_ds = mode_group.create_dataset(
            'points', shape=(n, n_samples, 2), dtype=np.float32)
        normals_ds = mode_group.create_dataset(
            'normals', shape=(n, n_samples, 2), dtype=np.float32)
        bar = IncrementalBar(max=n)
        for i in range(n):
            contours = image_to_contours(dataset[i][0])
            points, normals = sample_contours(contours, n_samples)
            points_ds[i] = points
            normals_ds[i] = normals
            bar.next()
        bar.finish()


def _save_data(group, n_samples):
    for mode in ('train', 'eval', 'test'):
        _save_mode_data(group, mode, n_samples)


def get_sampled_contour_path(n_samples):
    from dataset import data_dir
    folder = os.path.join(data_dir, 'sampled_contours')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return os.path.join(folder, '%d.hdf5' % n_samples)


def get_sampled_contour_dataset(mode, n_samples=2048):
    import h5py
    from dids.file_io.hdf5 import Hdf5Dataset, Hdf5ArrayDataset
    from dids.core import DictDataset
    path = get_sampled_contour_path(n_samples)
    if not os.path.isfile(path):
        with h5py.File(path, 'w') as group:
            _save_data(group, n_samples)

    root_ds = Hdf5Dataset(path, 'r')
    points_ds = Hdf5ArrayDataset(root_ds, '%s/points' % mode)
    normals_ds = Hdf5ArrayDataset(root_ds, '%s/normals' % mode)
    return DictDataset(points=points_ds, normals=normals_ds)
