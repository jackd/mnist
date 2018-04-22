import numpy as np
import matplotlib.pyplot as plt
from mnist.contour import get_sampled_contour_dataset
from mnist.dataset import get_dataset

mode = 'train'
n_samples = 2048


with get_dataset(mode) as image_ds:
    with get_sampled_contour_dataset(mode, n_samples) as sampled_ds:
        for key in image_ds:
            image = image_ds[key]
            sampled_group = sampled_ds[key]
            points, normals = (
                np.array(sampled_group[k]) for k in ('points', 'normals'))
            x = points[:, 1]
            y = points[:, 0]
            u = normals[:, 1]
            v = normals[:, 0]
            plt.plot(x, y, '*', color='b')
            plt.quiver(x, y, u, v, color='g')
            plt.xlim((0, 28))
            plt.ylim((0, 28))
            plt.show()
