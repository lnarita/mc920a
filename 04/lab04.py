import argparse
import os
from functools import reduce

import cv2 as cv
import numpy as np

MAX = 255
MIN = 0


def load_image(path):
    """
    loads the pixel matrix that represents the image stored in `path`
    :param path: the file path to the image
    :return: the np array representing that image
    """
    return cv.imread(path)


def convert_to_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def normalize_img(img, min, max):
    norm_img = np.zeros(np.shape(img))
    norm_img = cv.normalize(img, norm_img, min, max, cv.NORM_MINMAX)
    return norm_img


def save_img(img, name):
    cv.imwrite(name, img)


def extract_and_describe_with_brief(img, detector, descriptor):
    keypoints = detector.detect(img, None)
    return descriptor.compute(img, keypoints)


def create_visualization(img_a, img_b, keypoints_a, keypoints_b, matches, status):
    # initialize the output visualization image
    (hA, wA) = img_a.shape
    (hB, wB) = img_b.shape
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = img_a
    vis[0:hB, wA:] = img_b

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(keypoints_a[queryIdx][0]), int(keypoints_a[queryIdx][1]))
            ptB = (int(keypoints_b[trainIdx][0]) + wA, int(keypoints_b[trainIdx][1]))
            cv.line(vis, ptA, ptB, (0, 255, 0), 1)

    # return the visualization
    return vis


def stitch(img_a, img_b, ratio, describe, matcher, reproj_threshold, visualize_process=False):
    keypoints_a, features_a = describe(img_a)
    keypoints_b, features_b = describe(img_b)

    # (3)  compute similarities between descriptors
    matches = find_matches(features_a, features_b, matcher, ratio)
    if len(matches) > 4:
        # (4)  selecionar as melhores correspondˆencias para cada descritor de imagem.
        H, status = estimate_homography(keypoints_a, keypoints_b, matches, reproj_threshold)

        # (6)  aplicar uma proje ̧c ̃ao de perspectiva (cv2.warpPerspective) para alinhar as imagens.
        result = warp_img(H, img_a, img_b)

        if visualize_process:
            # (8)  desenhar retas entre pontos correspondentes no par de imagens.
            vis = create_visualization(img_a, img_b, keypoints_a, keypoints_b, matches, status)
            save_img(vis, "./imgs/out/stitch_proc.jpeg")
        return result
    return img_a


def estimate_homography(keypoints_a, keypoints_b, matches, reproj_threshold):
    points_a = np.array([np.array(keypoints_a[i].pt, dtype=np.float32) for (_, i) in matches], dtype=np.float32)
    points_b = np.array([np.array(keypoints_b[i].pt, dtype=np.float32) for (i, _) in matches], dtype=np.float32)

    # (5)  executar  a  técnica  RANSAC  (RANdom  SAmple  Consensus)  para  estimar  a  matriz  de  homografia(cv2.findHomography).
    (H, status) = cv.findHomography(points_a, points_b, cv.RANSAC, reproj_threshold)
    return H, status


def warp_img(H, img_a, img_b):
    result = cv.warpPerspective(img_a, H,
                                (img_a.shape[1] + img_b.shape[1], max(img_a.shape[0], img_b.shape[0])))

    # (7)  unir as imagens alinhadas e criar a imagem panorˆamica.
    result[0:img_b.shape[0], 0:img_b.shape[1]] = img_b
    return result


def find_matches(features_a, features_b, matcher, ratio):
    raw_matches = matcher.knnMatch(features_a, features_b, 2)
    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("format", metavar="INPUT_IMAGE_NAME_FORMAT",
                        help="image file name format")
    parser.add_argument("name", metavar="IMAGE_NAME",
                        help="image file name without extension")
    parser.add_argument("-n", metavar="FILE_COUNT", default=2, type=int,
                        help="image file name format")
    parser.add_argument("--input_path", default="./imgs",
                        help="input image file path")
    parser.add_argument("--output_path", default="./imgs/out",
                        help="output image path")
    parser.add_argument("--descriptor", choices=["sift", "surf", "orb", "brief", "kaze", "akaze"], default="sift",
                        help="feature descriptor")
    parser.add_argument("--detector", choices=["MSER", "FAST", "AGAST", "GFFT"], default="FAST",
                        help="feature detection")
    parser.add_argument("--matcher", choices=["bf", "flann"], default="bf",
                        help="feature matching")

    args = parser.parse_args()

    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_name_format = args.format
    img_name = args.name
    img_path = args.input_path
    img_count = args.n

    # (1) convert RGB image to grayscale
    imgs = [load_image("{}/{}.jpg".format(img_path, img_name_format.format(**{"i": i, "name": img_name}))).astype('uint8') for i in range(1, img_count + 1)]
    grayscale_imgs = [convert_to_grayscale(img) for img in imgs]
    normalized_imgs = [normalize_img(grayscale_img, MIN, MAX).astype(int) for grayscale_img in grayscale_imgs]

    # (2) detect keypoints and extract local invariant descriptors
    detector = {
        "MSER": lambda: cv.MSER_create(),
        "FAST": lambda: cv.FastFeatureDetector(),
        "AGAST": lambda: cv.AgastFeatureDetector(),
        "GFFT": lambda: cv.GFTTDetector()
    }[args.detector]()

    descriptor = {
        "sift": lambda: cv.xfeatures2d.SIFT_create(),
        "surf": lambda: cv.xfeatures2d.SURF_create(),
        "brief": lambda: cv.DescriptorExtractor_create("BRIEF"),
        "orb": lambda: cv.ORB_create(nfeatures=1500)
    }[args.descriptor]()


    def default_describe(img):
        return descriptor.detectAndCompute(img.astype('uint8'), None)


    descriptor_apply_function = {
        "brief": lambda img: extract_and_describe_with_brief(img, detector, descriptor)
    }.get(args.descriptor, lambda img: default_describe(img))

    flann_params = [{"algorithm": 0, "trees": 5}, {"checks": 50}]

    matcher = {
        "bf": cv.BFMatcher(),
        "flann": cv.FlannBasedMatcher(*flann_params)
    }[args.matcher]

    ratio = 0.75
    reproj_threshold = 4.
    result = reduce(lambda a, b: stitch(a, b, ratio, descriptor_apply_function, matcher, reproj_threshold, True), normalized_imgs)

    save_img(result, "{}/{}_result.jpeg".format(output_path, img_name))
