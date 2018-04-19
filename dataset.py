import os
from tensorflow.examples.tutorials.mnist import input_data
from dids.core import WrappedTupleDataset, ZippedDataset


def _get_data(mode):
    data = input_data.read_data_sets(
        os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'MNIST_data'),
        one_hot=False)
    if mode == 'train':
        return data.train
    elif mode == 'eval':
        return data.validation
    elif mode == 'test':
        return data.test
    else:
        raise ValueError('Unrecognized mode "%s"' % mode)


def _get_image_dataset(data):
    return WrappedTupleDataset(data.images.reshape(-1, 28, 28))


def _get_label_dataset(data):
    return WrappedTupleDataset(data.labels)


def get_dataset(mode):
    data = _get_data(mode)
    image_ds = _get_image_dataset(data)
    label_ds = _get_label_dataset(data)
    return ZippedDataset(image_ds, label_ds)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_points = 100
    dataset = get_dataset('train')
    for image, label in dataset.values():
        plt.imshow(image, cmap='gray')
        plt.title(str(label))
        plt.show()
