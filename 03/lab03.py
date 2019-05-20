import argparse
import os

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


def is_line(black_pixel_rate, white_to_black_transition_rate):
    return .1 < black_pixel_rate < .5 and white_to_black_transition_rate < .5


def is_word(black_pixel_rate, white_to_black_transition_rate):
    return .1 < black_pixel_rate < .57 and white_to_black_transition_rate < .486


def is_char(black_pixel_rate, white_to_black_transition_rate):
    return .1 < black_pixel_rate < .8 and white_to_black_transition_rate < .7


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCR?")
    # parser.add_argument("name", metavar="IMAGE_NAME",
    #                     help="image file name")
    # parser.add_argument("--out",
    #                     help="output image file prefix (which will be saved in the './imgs/out' folder, should not include the file extension")
    parser.add_argument("--seg", choices=["char", "word", "line", "line2"],
                        default="line2",
                        help="algorithm to execute. Defaults to 'all'")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(-1)

    seg = args.seg

    if not os.path.exists("./imgs/out"):
        os.makedirs("./imgs/out")
    if not os.path.exists("./imgs/out/{}".format(seg)):
        os.makedirs("./imgs/out/{}".format(seg))
    if not os.path.exists("./imgs/out/comp"):
        os.makedirs("./imgs/out/comp")
    if not os.path.exists("./imgs/out/comp/{}".format(seg)):
        os.makedirs("./imgs/out/comp/{}".format(seg))

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

    kernel_1 = {
        "char": np.array(np.ones((9, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((9, 12)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((1, 33)) * BLACK, dtype=np.uint8),
        "line2": np.array(np.ones((1, 100)) * BLACK, dtype=np.uint8)
    }[seg]

    img1 = cv.morphologyEx(img, cv.MORPH_DILATE, kernel_1)
    save_img(img1, "./imgs/out/{}/{}_step_1.pbm".format(seg, img_name))

    img2 = cv.morphologyEx(img1, cv.MORPH_ERODE, kernel_1)
    save_img(img2, "./imgs/out/{}/{}_step_2.pbm".format(seg, img_name))

    kernel_3 = {
        "char": np.array(np.ones((5, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((50, 1)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((100, 1)) * BLACK, dtype=np.uint8),
        "line2": np.array(np.ones((200, 1)) * BLACK, dtype=np.uint8)
    }[seg]

    img3 = cv.morphologyEx(img2, cv.MORPH_DILATE, kernel_3)
    save_img(img3, "./imgs/out/{}/{}_step_3.pbm".format(seg, img_name))

    img4 = cv.morphologyEx(img3, cv.MORPH_ERODE, kernel_3)
    save_img(img4, "./imgs/out/{}/{}_step_4.pbm".format(seg, img_name))

    img5 = cv.bitwise_and(img2, img4)
    save_img(img5, "./imgs/out/{}/{}_step_5.pbm".format(seg, img_name))

    kernel_6 = {
        "char": np.array(np.ones((1, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((5, 10)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((1, 15)) * BLACK, dtype=np.uint8),
        "line2": np.array(np.ones((1, 30)) * BLACK, dtype=np.uint8)
    }[seg]
    img6 = cv.morphologyEx(img5, cv.MORPH_CLOSE, kernel_6)
    save_img(img6, "./imgs/out/{}/{}_step_6.pbm".format(seg, img_name))

    img7 = np.copy(img6)
    num_comp, img7 = cv.connectedComponents(img6, img7, 8)
    print("Found {} connected components".format(num_comp))

    img7b = load_image("./imgs/out/{}/{}_step_6.pbm".format(seg, img_name))
    img7c = np.copy(img7b)
    contours, hierarchy = cv.findContours(img7b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    idx = 0
    text_cont = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv.boundingRect(cnt)

        roi = text_img[y:y + h, x:x + w]
        save_img(roi, "./imgs/out/comp/{}/{}_step_7_{}.pbm".format(seg, img_name, idx))
        roi_black_pixels = np.count_nonzero(roi == BLACK)
        # print("Black Pixels for idx {}: {}".format(idx, roi_black_pixels))
        black_pixels_rate = roi_black_pixels / (w * h)
        print("Black Pixels Rate for idx {}: {}".format(idx, black_pixels_rate))
        roi_v = np.count_nonzero(roi[:-1, :] - roi[1:, :] == BLACK)
        roi_h = np.count_nonzero(roi[:, :-1] - roi[:, 1:] == BLACK)
        # print("Vertical white to black transition for idx {}: {}".format(idx, roi_v))
        # print("Horizontal white to black transition for idx {}: {}".format(idx, roi_h))
        transition_rate = (roi_v + roi_h) / roi_black_pixels if roi_black_pixels != 0 else 0.
        print("Black Pixels Transition Rate for idx {}: {}".format(idx, transition_rate))

        # draw rectangle to visualize the bounding rect
        cv.rectangle(img7c, (x, y), (x + w, y + h), color=1, thickness=1)

        is_text = {
            "char": is_char,
            "word": is_word,
            "line": is_line,
            "line2": is_line
        }[seg]
        if is_text(black_pixels_rate, transition_rate):
            text_cont += 1
            cv.rectangle(text_img, (x, y), (x + w, y + h), color=1, thickness=1)
    save_img(img7c, "./imgs/out/{}/{}_step_7.pbm".format(seg, img_name))
    print("Found {} text parts".format(text_cont))

    black_pixels = np.where(text_img == BLACK)
    white_pixels = np.where(text_img == WHITE)
    text_img[black_pixels] = WHITE
    text_img[white_pixels] = BLACK
    save_img(text_img, "./imgs/out/{}/{}_text.pbm".format(seg, img_name))
