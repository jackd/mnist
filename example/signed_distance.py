import numpy as np
import matplotlib.pyplot as plt
from mnist.signed_distance import image_to_signed_distance
from mnist.dataset import get_dataset

n_points = 1000
dataset = get_dataset('train')
for image, label in dataset.values():
    sd = image_to_signed_distance(image)
    print(np.max(np.abs(sd)))
    print(np.min(sd), np.max(sd))

    sd /= np.max(np.abs(sd))
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.figure()
    plt.imshow(sd)
    plt.show()
