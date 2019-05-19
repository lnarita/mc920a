import os
import subprocess
import sys

import cv2 as cv
import numpy as np

WHITE = 0
BLACK = 1


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
    cv.imwrite(name, img)


def equiv_iterative(label, p, q):
    while label[p] != p:
        p = label[p]
        if p < q:
            temp = p
            p = q
            q = temp

    label[p] = q


if __name__ == '__main__':
    if not os.path.exists("./imgs/out"):
        os.makedirs("./imgs/out")
    if not os.path.exists("./imgs/out/comp"):
        os.makedirs("./imgs/out/comp")

    img_name = "bitmap"
    img_path = "./imgs/{}.pbm".format(img_name)
    img = load_image(img_path)
    img = normalize_img(img, WHITE, BLACK).astype(int)
    black_pixels = np.where(img == BLACK)
    white_pixels = np.where(img == WHITE)
    img[black_pixels] = WHITE
    img[white_pixels] = BLACK

    img = np.array(img, dtype=np.uint8)
    text_img = np.copy(img)

    kernel = np.array(np.ones((1, 100)) * BLACK, dtype=np.uint8)
    img1 = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
    save_img(img1, "./imgs/out/{}_step_1.pbm".format(img_name))

    img2 = cv.morphologyEx(img1, cv.MORPH_ERODE, kernel)
    save_img(img2, "./imgs/out/{}_step_2.pbm".format(img_name))

    kernel = np.array(np.ones((200, 1)) * BLACK, dtype=np.uint8)
    img3 = cv.morphologyEx(img2, cv.MORPH_DILATE, kernel)
    save_img(img3, "./imgs/out/{}_step_3.pbm".format(img_name))

    img4 = cv.morphologyEx(img3, cv.MORPH_ERODE, kernel)
    save_img(img4, "./imgs/out/{}_step_4.pbm".format(img_name))

    img5 = img2 & img4
    save_img(img5, "./imgs/out/{}_step_5.pbm".format(img_name))

    kernel = np.array(np.ones((1, 30)) * BLACK, dtype=np.uint8)
    img6 = cv.morphologyEx(img5, cv.MORPH_CLOSE, kernel)
    save_img(img6, "./imgs/out/{}_step_6.pbm".format(img_name))

    img7 = np.copy(img6)
    num_comp, img7 = cv.connectedComponents(img6, img7, 8)
    print("Found {} connected components".format(num_comp))

    img7b = load_image("./imgs/out/{}_step_6.pbm".format(img_name))
    img7c = np.copy(img7b)
    contours, hierarchy = cv.findContours(img7b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv.boundingRect(cnt)

        roi = img7c[y:y + h, x:x + w]
        save_img(roi, "./imgs/out/comp/{}_step_7_{}.pbm".format(img_name, idx))
        roi_black_pixels = np.count_nonzero(roi == 255)
        # print("Black Pixels for idx {}: {}".format(idx, roi_black_pixels))
        black_pixels_rate = roi_black_pixels / (w * h)
        print("Black Pixels Rate for idx {}: {}".format(idx, black_pixels_rate))
        roi_v = np.count_nonzero(roi[:-1, :] - roi[1:, :] == 255)
        roi_h = np.count_nonzero(roi[:, :-1] - roi[:, 1:] == 255)
        # print("Vertical white to black transition for idx {}: {}".format(idx, roi_v))
        # print("Horizontal white to black transition for idx {}: {}".format(idx, roi_h))
        transition_rate = (roi_v + roi_h) / roi_black_pixels
        print("Black Pixels Transition Rate for idx {}: {}".format(idx, transition_rate))

        # draw a green rectangle to visualize the bounding rect
        cv.rectangle(img7c, (x, y), (x + w, y + h), color=1, thickness=1)

        if .4 < black_pixels_rate < .95 and transition_rate < .35:
            cv.rectangle(text_img, (x, y), (x + w, y + h), color=1, thickness=1)
    save_img(img7c, "./imgs/out/{}_step_7.pbm".format(img_name))
    save_img(text_img, "./imgs/out/{}_text.pbm".format(img_name))
