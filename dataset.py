import os
from tensorflow.examples.tutorials.mnist import input_data
from dids.core import WrappedListDataset, ZippedDataset

data_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'MNIST_data')


def _get_data(mode):
    data = input_data.read_data_sets(data_dir, one_hot=False)
    if mode == 'train':
        return data.train
    elif mode == 'eval':
        return data.validation
    elif mode == 'test':
        return data.test
    else:
        raise ValueError('Unrecognized mode "%s"' % mode)


def _get_image_dataset(data):
    return WrappedListDataset(data.images.reshape(-1, 28, 28))


def _get_label_dataset(data):
    return WrappedListDataset(data.labels)


def get_dataset(mode):
    data = _get_data(mode)
    image_ds = _get_image_dataset(data)
    label_ds = _get_label_dataset(data)
    return ZippedDataset(image_ds, label_ds)


_lengths = {'train': 55000, 'eval': 5000, 'test': 10000}
image_shape = (28, 28)


def get_dataset_length(mode):
    return _lengths[mode]


def _get_dataset_length(mode):
    with get_dataset(mode) as ds:
        return len(ds)


if __name__ == '__main__':
    # for mode in ('train', 'eval', 'test'):
    #     dataset = get_dataset(mode)
    #     with dataset:
    #         print(mode, len(dataset))

    import matplotlib.pyplot as plt
    n_points = 100
    dataset = get_dataset('train')
    for image, label in dataset.values():
        plt.imshow(image, cmap='gray')
        plt.title(str(label))
        plt.show()
