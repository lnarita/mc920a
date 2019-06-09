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


def random_color():
    return np.random.choice(range(256), size=3).astype(np.int, copy=False)


def create_visualization(img_a, img_b, keypoints_a, keypoints_b, matches, status):
    # initialize the output visualization image
    (hA, wA) = img_a.shape
    (hB, wB) = img_b.shape
    vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
    vis[0:hA, 0:wA] = img_a
    vis[0:hB, wA:] = img_b
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2RGB)

    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully matched
        if s == 1:
            # draw the match
            ptA = (int(keypoints_a[queryIdx].pt[0]), int(keypoints_a[queryIdx].pt[1]))
            ptB = (int(keypoints_b[trainIdx].pt[0]) + wA, int(keypoints_b[trainIdx].pt[1]))
            color = random_color()
            cv.line(vis, ptA, ptB, (int(color[0]), int(color[1]), int(color[2])), 1)

    # return the visualization
    return vis


def stitch(img_a, img_b, ratio, describe, matcher, reproj_threshold, visualize_process, idx_a, idx_b, img_name, algorithm):
    keypoints_a, features_a = describe(img_a)
    keypoints_b, features_b = describe(img_b)

    # (3) compute similarities between descriptors
    matches = find_matches(features_a, features_b, matcher, ratio)
    if len(matches) > 4:
        # (4) select matches
        H, status = estimate_homography(keypoints_a, keypoints_b, matches, reproj_threshold)

        if H is not None:
            # (6) warp and align images
            result = warp_img(H, img_a, img_b)

            if visualize_process:
                # (8) draw lines connecting corresponding points
                vis = create_visualization(img_a, img_b, keypoints_a, keypoints_b, matches, status)
                save_img(vis, "./imgs/out/stitch_proc_{}_{}_{}_{}.jpg".format(img_name, idx_a, idx_b, algorithm))
                save_img(result, "./imgs/out/partial_result_{}_{}_{}_{}.jpg".format(img_name, idx_a, idx_b, algorithm))
            return -idx_a, result
        else:
            return idx_a, img_a
    return idx_a, img_a


def estimate_homography(keypoints_a, keypoints_b, matches, reproj_threshold):
    points_a = np.array([np.array(keypoints_a[i].pt, dtype=np.float32) for (_, i) in matches], dtype=np.float32)
    points_b = np.array([np.array(keypoints_b[i].pt, dtype=np.float32) for (i, _) in matches], dtype=np.float32)

    # (5) RANSAC
    (H, status) = cv.findHomography(points_a, points_b, cv.RANSAC, reproj_threshold)
    return H, status


def warp_img(H, img_a, img_b):
    result = cv.warpPerspective(np.array(img_a, dtype="uint8"), H,
                                (img_a.shape[1] + img_b.shape[1], max(img_a.shape[0], img_b.shape[0])))

    # (7) stitch images together
    result[0:img_b.shape[0], 0:img_b.shape[1]] = img_b
    return result


def find_matches(features_a, features_b, matcher, ratio):
    raw_matches = matcher.knnMatch(features_a, features_b, 2)
    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    return matches


def create_panorama(descriptor_name, detector_name, normalized_imgs, output_path, img_name):
    # (2) detect keypoints and extract local invariant descriptors
    detector = {
        "MSER": lambda: cv.MSER_create(),
        "FAST": lambda: cv.FastFeatureDetector_create(),
        "AGAST": lambda: cv.AgastFeatureDetector_create(),
        "GFFT": lambda: cv.GFTTDetector_create(),
        "STAR": lambda: cv.xfeatures2d.StarDetector_create()
    }[detector_name]()
    descriptor = {
        "sift": lambda: cv.xfeatures2d.SIFT_create(),
        "surf": lambda: cv.xfeatures2d.SURF_create(),
        "brief": lambda: cv.xfeatures2d.BriefDescriptorExtractor_create(),
        "orb": lambda: cv.ORB_create(nfeatures=1500),
        "kaze": lambda: cv.KAZE_create(),
        "akaze": lambda: cv.AKAZE_create(),
    }[descriptor_name]()

    def default_describe(img):
        return descriptor.detectAndCompute(img.astype('uint8'), None)

    descriptor_apply_function = {
        "brief": lambda img: extract_and_describe_with_brief(img, detector, descriptor)
    }.get(descriptor_name, lambda img: default_describe(img))
    matcher = cv.BFMatcher()
    ratio = 0.75
    reproj_threshold = 4.
    alg = descriptor_name if descriptor_name in ["sift", "surf", "orb", "kaze", "akaze"] else "{}_{}".format(descriptor_name,
                                                                                                             detector_name)

    def r_stitch(a, b):
        return stitch(a[1], b[1], ratio, descriptor_apply_function, matcher, reproj_threshold, True, a[0] + 1, b[0] + 1, img_name,
                      alg)

    result = reduce(r_stitch, enumerate(normalized_imgs))
    save_img(result[1], "{}/{}_result_{}.jpg".format(output_path, img_name, alg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("format", metavar="INPUT_IMAGE_NAME_FORMAT",
                        help="image file name format")
    parser.add_argument("name", metavar="IMAGE_NAME",
                        help="image file name without extension")
    parser.add_argument("-n", metavar="FILE_COUNT", default=2, type=int,
                        help="number of images to stitch")
    parser.add_argument("--input_path", default="./imgs",
                        help="input image file path")
    parser.add_argument("--output_path", default="./imgs/out",
                        help="output image path")
    parser.add_argument("--descriptor", choices=["sift", "surf", "orb", "brief", "kaze", "akaze", "all"], default="sift",
                        help="feature descriptor")
    parser.add_argument("--detector", choices=["MSER", "FAST", "AGAST", "GFFT", "STAR"], default="FAST",
                        help="feature detection")

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
    normalized_imgs = [normalize_img(grayscale_img, MIN, MAX).astype(np.uint8) for grayscale_img in grayscale_imgs]

    if args.descriptor == "all":
        for dscrpt in ["orb", "brief", "kaze", "akaze", "sift", "surf"]:
            for dtctr in ["MSER", "FAST", "AGAST", "GFFT", "STAR"] if dscrpt == "brief" else ["FAST"]:
                create_panorama(dscrpt, dtctr, normalized_imgs, output_path, img_name)
    else:
        create_panorama(args.descriptor, args.detector, normalized_imgs, output_path, img_name)
