import argparse
import functools
import multiprocessing
import os
import timeit

import cv2
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def load_image(path):
    """
    loads the pixel matrix that represents the image stored in `path`
    :param path: the file path to the image
    :return: the np array representing that image
    """
    return cv2.imread(path)


def save_img(img, file_name):
    cv2.imwrite(file_name, img)
    print("saved output to {}".format(file_name))


def flatten(img):
    return img.reshape(-1, 3)


def unflatten(img, shape):
    return img.reshape(shape)


class Codebook(dict):
    __slots__ = ['labels', 'cluster_centers']

    def __init__(self, labels, cluster_centers):
        super().__init__()
        self.labels = labels
        self.cluster_centers = cluster_centers


def timed(func):
    @functools.wraps(func)
    def wrapper(*args):
        timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        ret_val = {stmt}
    _t1 = _timer()
    return _t1 - _t0, ret_val
        """
        t = timeit.Timer(lambda: func(*args))
        elapsed, result = t.timeit(number=1)
        print("{} process time: {:.2f} seconds".format(func.__name__, elapsed))
        return result

    return wrapper


@timed
def cluster(image, k, n_init=10, max_iter=300):
    """
    Groups the image's pixels into clusters by their colour similarity and
    then creates an image representation using only the n_clusters colours
    """
    k_colours = KMeans(k, n_init=n_init, max_iter=max_iter).fit(image)
    compressed = k_colours.cluster_centers_[k_colours.labels_]
    compressed = np.reshape(compressed, image.shape)
    return compressed, Codebook(k_colours.labels_, k_colours.cluster_centers_)


@timed
def extract_colour_components(flat_img):
    b = flat_img[:, 0]
    g = flat_img[:, 1]
    r = flat_img[:, 2]
    return b, g, r


@timed
def extract_unique_colours(flat_img):
    return np.vstack([tuple(r) for r in flat_img])


@timed
def convert_bgr_to_rgb(b, g, r):
    rgb = np.zeros((b.shape[0], 3))
    rgb[:, 0] = r
    rgb[:, 1] = g
    rgb[:, 2] = b
    return rgb


@timed
def plot_colour_matrix(colours, file_name):
    b, g, r = extract_colour_components(colours)

    rgb = convert_bgr_to_rgb(b, g, r)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r * 255, g * 255, b * 255, c=rgb)
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        print("saved output to {}".format(file_name))


@timed
def plot_clusters(flat_img, labels, colours, file_name):
    """
    Plots the original image's pixels with their cluster colours
    """
    b, g, r = extract_colour_components(flat_img)

    rgb = convert_bgr_to_rgb(*extract_colour_components(colours))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r * 255, g * 255, b * 255, c=rgb[labels])
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')


def plot_alternate_colour_matrix(a, b, c, colours, file_name, x_label, y_label, z_label):
    rgb = convert_bgr_to_rgb(*extract_colour_components(colours))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(a, b, c, c=rgb)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        print("saved output to {}".format(file_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("name", metavar="IMAGE_NAME",
                        help="image file name without extension")
    parser.add_argument("-k", default=128, type=int,
                        help="k")
    parser.add_argument("-ext", default="png", choices=["png", "jpg"],
                        help="file extension")
    parser.add_argument("--input_path", default="./imgs",
                        help="input image file path")
    parser.add_argument("--output_path", default="./imgs/out",
                        help="output image path")

    args = parser.parse_args()

    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_name = args.name
    img_path = args.input_path
    k = args.k
    ext = args.ext

    image = load_image("{}/{}.{}".format(img_path, img_name, ext))
    flat_image = flatten(image)
    normalized_img = flat_image / 255.

    k_image, codebook = cluster(normalized_img, k)

    printable_k_image = np.array(k_image * 255., dtype=np.int32)
    printable_k_image = unflatten(printable_k_image, image.shape)

    save_img(printable_k_image, "{}/{}_{}.{}".format(output_path, img_name, k, ext))

    def plot_hsv():
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        flat_hsv = flatten(hsv)
        h, s, v = extract_colour_components(flat_hsv)
        plot_alternate_colour_matrix(h * 2., s / 255., v / 255., normalized_img, "{}/{}_hsv.{}".format(output_path, img_name, ext), "H", "S", "V")


    def plot_hls():
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        flat_hls = flatten(hls)
        h, l, s = extract_colour_components(flat_hls)
        plot_alternate_colour_matrix(h * 2., l / 255., s / 255., normalized_img, "{}/{}_hls.{}".format(output_path, img_name, ext), "H", "L", "S")


    def plot_original_colours():
        original_colours = extract_unique_colours(normalized_img)
        plot_colour_matrix(original_colours, "{}/{}_scatter.{}".format(output_path, img_name, ext))


    def plot_cluster_colours():
        cluster_colours = extract_unique_colours(k_image)
        plot_colour_matrix(cluster_colours, "{}/{}_{}_scatter.{}".format(output_path, img_name, k, ext))


    def plot_pixel_clusters():
        plot_clusters(normalized_img, codebook.labels, codebook.cluster_centers, "{}/{}_{}_clusters.{}".format(output_path, img_name, k, ext))


    pool = multiprocessing.Pool(5)

    plots = [
        pool.apply_async(func=plot_hls),
        pool.apply_async(func=plot_hsv),
        pool.apply_async(func=plot_original_colours),
        pool.apply_async(func=plot_cluster_colours),
        pool.apply_async(func=plot_pixel_clusters),
    ]

    pool.close()
    pool.join()

    for p in plots:
        p.get(timeout=10)

    exit(0)
