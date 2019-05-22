import argparse
import os

import cv2 as cv
import numpy as np
import pytesseract as ocr
from pytesseract.pytesseract import run_tesseract

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
    return .1 < black_pixel_rate < .5 and white_to_black_transition_rate > .2


def is_word(black_pixel_rate, white_to_black_transition_rate):
    return .1 < black_pixel_rate < .6 and white_to_black_transition_rate > .2


def is_char(black_pixel_rate, white_to_black_transition_rate):
    return .1 < black_pixel_rate < .8 and white_to_black_transition_rate < .7


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("name", metavar="IMAGE_NAME",
                        help="image file name without extension")
    parser.add_argument("--p", default="./imgs",
                        help="input image file path")
    parser.add_argument("--op", default="./imgs/out",
                        help="output image path")
    parser.add_argument("--out",
                        help="output image file prefix")
    parser.add_argument("--ocr", action="store_true",
                        help="use tesseract to try to parse identified text segments")
    parser.add_argument("--lang", default="eng",
                        help="text language. Only makes sense when using --ocr")
    parser.add_argument("--seg", choices=["char", "word", "line"], default="line",
                        help="search for `line`s, `char`s or `word`s")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(-1)

    seg = args.seg
    output_path = args.op

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists("{}/{}".format(output_path, seg)):
        os.makedirs("{}/{}".format(output_path, seg))
    if not os.path.exists("{}/comp".format(output_path)):
        os.makedirs("{}/comp".format(output_path))
    if not os.path.exists("{}/comp/{}".format(output_path, seg)):
        os.makedirs("{}/comp/{}".format(output_path, seg))
    if not os.path.exists("{}/{}/tess".format(output_path, seg)):
        os.makedirs("{}/{}/tess".format(output_path, seg))
        os.makedirs("{}/{}/tess/text".format(output_path, seg))

    img_name = args.name
    img_path = args.p
    if not args.out:
        args.out = img_name
    img_out_name = args.out

    img = load_image("{}/{}.pbm".format(img_path, img_name))
    img = normalize_img(img, WHITE, BLACK).astype(int)
    black_pixels = np.where(img == BLACK)
    white_pixels = np.where(img == WHITE)
    img[black_pixels] = WHITE
    img[white_pixels] = BLACK

    img = np.array(img, dtype=np.uint8)
    text_img = np.copy(img)

    kernel_1 = {
        "char": np.array(np.ones((9, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((7, 12)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((1, 100)) * BLACK, dtype=np.uint8)
    }[seg]

    img1 = cv.morphologyEx(img, cv.MORPH_DILATE, kernel_1)
    save_img(img1, "{}/{}/{}_step_1.pbm".format(output_path, seg, img_out_name))

    img2 = cv.morphologyEx(img1, cv.MORPH_ERODE, kernel_1)
    save_img(img2, "{}/{}/{}_step_2.pbm".format(output_path, seg, img_out_name))

    kernel_3 = {
        "char": np.array(np.ones((5, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((50, 1)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((200, 1)) * BLACK, dtype=np.uint8)
    }[seg]

    img3 = cv.morphologyEx(img, cv.MORPH_DILATE, kernel_3)
    save_img(img3, "{}/{}/{}_step_3.pbm".format(output_path, seg, img_out_name))

    img4 = cv.morphologyEx(img3, cv.MORPH_ERODE, kernel_3)
    save_img(img4, "{}/{}/{}_step_4.pbm".format(output_path, seg, img_out_name))

    img5 = cv.bitwise_and(img2, img4)
    save_img(img5, "{}/{}/{}_step_5.pbm".format(output_path, seg, img_out_name))

    kernel_6 = {
        "char": np.array(np.ones((1, 1)) * BLACK, dtype=np.uint8),
        "word": np.array(np.ones((5, 10)) * BLACK, dtype=np.uint8),
        "line": np.array(np.ones((1, 30)) * BLACK, dtype=np.uint8)
    }[seg]
    img6 = cv.morphologyEx(img5, cv.MORPH_CLOSE, kernel_6)
    save_img(img6, "{}/{}/{}_step_6.pbm".format(output_path, seg, img_out_name))

    img7 = np.copy(img6)
    num_comp, img7 = cv.connectedComponents(img6, img7, 8)
    print("Found {} connected components".format(num_comp))

    img7b = load_image("{}/{}/{}_step_6.pbm".format(output_path, seg, img_out_name))
    img7c = np.copy(img7b)
    contours, hierarchy = cv.findContours(img7b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    idx = 0
    text_cont = 0
    is_text = {
        "char": is_char,
        "word": is_word,
        "line": is_line
    }[seg]
    tesseract_config = {
        "char": "--psm 10",
        "word": "--psm 8",
        "line": "--psm 13"
    }[seg]
    for cnt in contours:
        idx += 1
        x, y, w, h = cv.boundingRect(cnt)

        roi = text_img[y:y + h, x:x + w]
        save_img(roi, "{}/comp/{}/{}_step_7_{}.pbm".format(output_path, seg, img_out_name, idx))
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
        if is_text(black_pixels_rate, transition_rate):
            if args.ocr:
                cv.putText(text_img, str(idx),
                           (x, y), color=1, fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)
                padding_y, padding_x = y - 5 if y > 5 else y, x - 5 if x > 5 else x
                save_img(normalize_img(img[padding_y:y + h, padding_x:x + w], 0, 255), "{}/{}/tess/{}__{}.png".format(output_path, seg, img_out_name, idx))
                run_tesseract(input_filename="{}/{}/tess/{}__{}.png".format(output_path, seg, img_out_name, idx),
                              output_filename_base="{}/{}/tess/text/{}__{}".format(output_path, seg, img_out_name, idx),
                              extension="png",
                              lang=args.lang,
                              config=tesseract_config,
                              nice=0)
            text_cont += 1
            cv.rectangle(text_img, (x, y), (x + w, y + h), color=1, thickness=1)
    save_img(img7c, "{}/{}/{}_step_7.pbm".format(output_path, seg, img_out_name))
    print("Found {} text parts".format(text_cont))

    black_pixels = np.where(text_img == BLACK)
    white_pixels = np.where(text_img == WHITE)
    text_img[black_pixels] = WHITE
    text_img[white_pixels] = BLACK
    save_img(text_img, "{}/{}/{}_text.pbm".format(output_path, seg, img_out_name))
