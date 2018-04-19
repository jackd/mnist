import matplotlib.pyplot as plt
from mnist.cloud import sample_image, image_to_cloud
from mnist.dataset import get_dataset

n_points = 100
dataset = get_dataset('train')
for image, label in dataset.values():
    y, x = image_to_cloud(image)
    sy, sx = sample_image(image, n_points)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(str(label))
    plt.figure()
    plt.plot(x, y, '*')
    plt.xlim((0, 28))
    plt.ylim((28, 0))
    plt.figure()
    plt.plot(sx, sy, '*')
    plt.xlim((0, 28))
    plt.ylim((28, 0))
    plt.show()
