# coding: utf-8
import argparse
import os

import cv2 as cv
import numpy as np

from hilbertcurve import HilbertCurve


def load_image(path):
    """
    loads the pixel matrix that represents the image stored in `path`
    :param path: the file path to the image
    :return: the np array representing that image
    """
    return cv.imread(path, 0)


def normalize_img(img, min, max):
    norm_img = np.zeros(np.shape(img))
    norm_img = cv.normalize(img, norm_img, min, max, cv.NORM_MINMAX)
    return norm_img


def save_img(img, name):
    cv.imwrite("./imgs/out/{}.pgm".format(name), img)


WHITE = 0
BLACK = 255

dithering_matrix = np.array([[6, 8, 4],
                             [1, 0, 3],
                             [5, 2, 7]])

bayer2 = np.array([[0, 2], [3, 1]])

bayer4 = np.array([[0, 12, 3, 15],
                   [8, 4, 11, 7],
                   [2, 14, 1, 13],
                   [10, 6, 9, 5]])

bayer8 = np.array([[0, 48, 12, 60, 3, 51, 15, 63],
                   [32, 16, 44, 28, 35, 19, 47, 31],
                   [8, 56, 4, 52, 11, 59, 7, 55],
                   [40, 24, 36, 20, 43, 27, 39, 23],
                   [2, 50, 14, 62, 1, 49, 13, 61],
                   [34, 18, 46, 30, 33, 17, 45, 29],
                   [10, 58, 6, 54, 9, 57, 5, 53],
                   [42, 26, 38, 22, 41, 25, 37, 21]])

clusterdot4 = np.array([[12, 5, 6, 13],
                        [4, 0, 1, 7],
                        [11, 3, 2, 8],
                        [15, 10, 9, 14],
                        ])

clusterdot8 = np.array([[24, 10, 12, 26, 35, 47, 49, 37],
                        [8, 0, 2, 14, 45, 59, 61, 51],
                        [22, 6, 4, 16, 43, 57, 63, 53],
                        [30, 20, 18, 28, 33, 41, 55, 39],
                        [34, 46, 48, 36, 25, 11, 13, 27],
                        [44, 58, 60, 50, 9, 1, 3, 15],
                        [42, 56, 62, 52, 23, 7, 5, 17],
                        [32, 40, 54, 38, 31, 21, 19, 29]])


def dither(img, dither_mask):
    img_height, img_width = img.shape
    mask_height, mask_width = dither_mask.shape
    dithered_img = np.ones((img_height * mask_height, img_width * mask_width)) * WHITE
    normalized_img = normalize_img(img, 0, mask_height * mask_width).astype(int)
    for y in range(img_height):
        for x in range(img_width):
            pixel_value = normalized_img[y, x]
            # get indexes that should be black
            relative_y, relative_x = np.where(dither_mask < pixel_value)
            # calculate new pixels position in the resulting image
            dither_idx_x, dither_idx_y = x * mask_width, y * mask_height
            relative_x += dither_idx_x
            relative_y += dither_idx_y
            # paint black pixels
            dithered_img[(relative_y, relative_x)] = BLACK
    return dithered_img


def floyd_steinberg_unidirectional(img):
    img_height, img_width = img.shape
    dithered_img = np.copy(img)
    for y in range(img_height):
        for x in range(img_width):
            pixel_value = dithered_img[y, x]
            new_value = BLACK if pixel_value > 127 else WHITE
            dithered_img[y, x] = new_value
            error = pixel_value - new_value
            if x < img_width - 1:
                dithered_img[y, x + 1] += round(error * 7 / 16)
            if x > 0 and y < img_height - 1:
                dithered_img[y + 1, x - 1] += round(error * 3 / 16)
            if y < img_height - 1:
                dithered_img[y + 1, x] += round(error * 5 / 16)
            if x < img_width - 1 and y < img_height - 1:
                dithered_img[y + 1, x + 1] += round(error * 1 / 16)
    dithered_img[np.where(dithered_img < WHITE)] = WHITE  # remove all negative values
    dithered_img[np.where(dithered_img > BLACK)] = BLACK  # remove all values up our limit

    return dithered_img


def floyd_steinberg_alternate(img):
    img_height, img_width = img.shape
    dithered_img = np.copy(img)
    for y in range(img_height):
        for x in range(img_width):
            if y % 2 == 0:  # >
                pixel_value = dithered_img[y, x]
                new_value = BLACK if pixel_value > 127 else WHITE
                dithered_img[y, x] = new_value
                error = pixel_value - new_value
                if x < img_width - 1:
                    dithered_img[y, x + 1] += round(error * 7 / 16)
                if x > 0 and y < img_height - 1:
                    dithered_img[y + 1, x - 1] += round(error * 3 / 16)
                if y < img_height - 1:
                    dithered_img[y + 1, x] += round(error * 5 / 16)
                if x < img_width - 1 and y < img_height - 1:
                    dithered_img[y + 1, x + 1] += round(error * 1 / 16)
            else:  # <
                new_x = img_width - 1 - x
                pixel_value = dithered_img[y, new_x]
                new_value = BLACK if pixel_value > 127 else WHITE
                dithered_img[y, new_x] = new_value
                error = pixel_value - new_value
                if new_x > 0:
                    dithered_img[y, new_x - 1] += round(error * 7 / 16)
                if new_x > 0 and y < img_height - 1:
                    dithered_img[y + 1, new_x - 1] += round(error * 1 / 16)
                if y < img_height - 1:
                    dithered_img[y + 1, new_x] += round(error * 5 / 16)
                if new_x < img_width - 1 and y < img_height - 1:
                    dithered_img[y + 1, new_x + 1] += round(error * 3 / 16)
    dithered_img[np.where(dithered_img < WHITE)] = WHITE  # remove all negative values
    dithered_img[np.where(dithered_img > BLACK)] = BLACK  # remove all values up our limit

    return dithered_img


def floyd_steinberg_hilbert_curve(img):
    img_height, img_width = img.shape
    dithered_img = np.copy(img)
    curve = HilbertCurve(np.ceil(np.log2(img_height)).astype(int), 2)
    already_visited = np.zeros(img.shape)
    last_x, x = 0, 0
    last_y, y = 0, 0
    for i in range(curve.max_h):
        last_x = x  # use last pixel position to determine if we're going up, down, left or right to propagate the error to the right neighbors
        last_y = y
        y, x = curve.coordinates_from_distance(i)
        y = img_height - 1 - y
        if not (-1 < x < img_width and -1 < y < img_height):
            # check image bounds, if current pixel is out of image, do nothing and go to next pixel
            # the image is probably not a square, or its dimensions are not 2^n
            continue
        pixel_value = dithered_img[y, x]
        already_visited[y, x] = 1.
        new_value = BLACK if pixel_value > 127 else WHITE
        dithered_img[y, x] = new_value
        error = pixel_value - new_value
        up, down, left, right = y - 1, y + 1, x - 1, x + 1
        up_left, up_right, down_left, down_right = (up, left), (up, right), (down, left), (down, right)
        if last_x <= x:  # going right
            if x < img_width - 1 and already_visited[y, right] == 0.:
                dithered_img[y, right] += round(error * 7 / 16)
            if last_y < y:  # going down
                if x > 0 and y < img_height - 1 and already_visited[down_left] == 0.:
                    dithered_img[down_left] += round(error * 3 / 16)
                if y < img_height - 1 and already_visited[down, x] == 0.:
                    dithered_img[down, x] += round(error * 5 / 16)
                if x < img_width - 1 and y < img_height - 1 and already_visited[down_right] == 0.:
                    dithered_img[down_right] += round(error * 1 / 16)
            else:  # going up
                if x > 0 and y > 0 and already_visited[up_left] == 0.:
                    dithered_img[up_left] += round(error * 3 / 16)
                if y > 0 and already_visited[up, x] == 0.:
                    dithered_img[up, x] += round(error * 5 / 16)
                if x < img_width - 1 and y > 0 and already_visited[up_right] == 0.:
                    dithered_img[up_right] += round(error * 1 / 16)
        else:  # going left
            if x > 0 and already_visited[y, left] == 0.:
                dithered_img[y, left] += round(error * 7 / 16)
            if last_y < y:  # going down
                if x < img_width - 1 and y < img_height - 1 and already_visited[down_right] == 0.:
                    dithered_img[down_right] += round(error * 3 / 16)
                if y < img_height - 1 and already_visited[down, x] == 0.:
                    dithered_img[down, x] += round(error * 5 / 16)
                if x > 0 and y < img_height - 1 and already_visited[down_left] == 0.:
                    dithered_img[down_left] += round(error * 1 / 16)
            else:  # going up
                if x < img_width - 1 and y > 0 and already_visited[up_right] == 0.:
                    dithered_img[up_right] += round(error * 3 / 16)
                if y > 0 and already_visited[up, x] == 0.:
                    dithered_img[up, x] += round(error * 5 / 16)
                if x > 0 and y > 0 and already_visited[up_left] == 0.:
                    dithered_img[up_left] += round(error * 1 / 16)
    # for some reason, this Hilbert Curve impl. don't visit the last position
    dithered_img[0, 0] = BLACK if dithered_img[0, 0] > dithered_img[0, 0] else WHITE
    dithered_img[np.where(dithered_img < WHITE)] = WHITE  # remove all negative values
    dithered_img[np.where(dithered_img > BLACK)] = BLACK  # remove all values up our limit

    return dithered_img


def do_all(img):
    result = {"3x3": dither(img, dithering_matrix),
              "bayer2": dither(img, bayer2),
              "bayer4": dither(img, bayer4),
              "bayer8": dither(img, bayer8),
              "cluster_dot4": dither(img, clusterdot4),
              "cluster_dot8": dither(img, clusterdot8),
              "fs_uni": floyd_steinberg_unidirectional(img),
              "fs_alt": floyd_steinberg_alternate(img),
              "fs_hil": floyd_steinberg_hilbert_curve(img)}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Some dithering algorithms implementations")
    parser.add_argument("name", metavar="IMAGE_NAME",
                        help="image file name (which must be inside the './imgs/in' folder, should not include the file extension")
    parser.add_argument("--out",
                        help="output image file prefix (which will be saved in the './imgs/out' folder, should not include the file extension")
    parser.add_argument("--alg", choices=["simple3", "bayer2", "bayer4", "bayer8", "cluster_dot8", "fs_uni", "fs_alt", "fs_hil", "all"],
                        default="all",
                        help="algorithm to execute. Defaults to 'all'")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(-1)

    if not os.path.exists("./imgs/out"):
        os.makedirs("./imgs/out")

    if not args.out:
        args.out = args.name

    image = load_image("./imgs/in/{}.pgm".format(args.name))

    result = {
        "simple3": lambda img: dither(img, dithering_matrix),
        "bayer2": lambda img: dither(img, bayer2),
        "bayer4": lambda img: dither(img, bayer4),
        "bayer8": lambda img: dither(img, bayer8),
        "cluster_dot4": lambda img: dither(img, clusterdot4),
        "cluster_dot8": lambda img: dither(img, clusterdot8),
        "fs_uni": lambda img: floyd_steinberg_unidirectional(img),
        "fs_alt": lambda img: floyd_steinberg_alternate(img),
        "fs_hil": lambda img: floyd_steinberg_hilbert_curve(img),
        "all": lambda img: do_all(img)
    }[args.alg](image)

    if isinstance(result, dict):
        for k in result.keys():
            save_img(result[k], "{}_{}".format(args.out, k))
    else:
        save_img(result, "{}_{}".format(args.out, args.alg))
