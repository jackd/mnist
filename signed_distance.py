"""
Requires skmm

pip install scikit-fmm
"""
import os
import numpy as np


def image_to_signed_distance(image):
    import skfmm
    return skfmm.distance(image*2 - 1) / max(*image.shape)


def _save_mode_data(group, mode):
    from dataset import get_dataset
    from progress.bar import IncrementalBar
    dataset = get_dataset(mode)
    n = len(dataset)
    with dataset:
        data = group.create_dataset(mode, shape=(n, 28, 28), dtype=np.float32)
        bar = IncrementalBar(max=n)
        for i in range(n):
            data[i] = image_to_signed_distance(dataset[i][0])
            bar.next()
        bar.finish()


def _save_data(group):
    for mode in ('train', 'eval', 'test'):
        _save_mode_data(group, mode)


def get_signed_distance_path():
    from dataset import data_dir
    return os.path.join(data_dir, 'signed_distance.hdf5')


def get_signed_distance_dataset(mode):
    import h5py
    from dids.core import WrappedListDataset
    path = get_signed_distance_path()
    if not os.path.isfile(path):
        with h5py.File(path, 'w') as group:
            _save_data(group)

    with h5py.File(path) as group:
        data = np.array(group[mode])
    return WrappedListDataset(data)
    # return _DatasetProvider.get_dataset(mode).map(lambda x: np.array(x))


# class _DatasetProvider(object):
#     def __init__(self):
#         raise RuntimeError('_DatasetProvider not meant to be instantiated.')
#
#     path = get_signed_distance_path()
#     _parent = None
#
#     @staticmethod
#     def parent(self):
#         if _DatasetProvider._parent is None:
#             _DatasetProvider._parent = Hdf5Dataset(path, 'r')
#         return _DatasetProvider._parent
#
#     @staticmethod
#     def get_dataset(mode):
#         from dids.file_io.hdf5 import Hdf5Dataset, Hdf5ArrayDataset
#         import h5py
#         path = _DatasetProvider.path
#         if not os.path.isfile(path):
#             with h5py.File(path, 'w') as group:
#                 _save_data(group)
#
#         return Hdf5ArrayDataset(_DatasetProvider.parent(), mode)
