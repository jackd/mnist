import matplotlib.pyplot as plt
from mnist.dataset import get_dataset
from mnist.contour import image_to_contours
from mnist.contour import sample_contours

n_points = 1000
dataset = get_dataset('train')
for image, label in dataset.values():
    contours = image_to_contours(image)
    points, normals = sample_contours(contours, n_points)
    plt.figure()
    plt.imshow(image, cmap='gray')
    x = points[:, 1]
    y = points[:, 0]
    u = normals[:, 1]
    v = normals[:, 0]
    plt.plot(x, y, '*', color='b')
    plt.quiver(x, y, u, v, color='g')
    for contour in contours:
        plt.plot(
            contour[:, 1], contour[:, 0], color='red', linestyle='dotted')
    plt.xlim((0, 28))
    plt.ylim((0, 28))
    plt.show()
