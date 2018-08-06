#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import argparse


def combine(oimage, logo_path, eimage, scale):
    logo = cv2.imread(logo_path)
    logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST)
    logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, logo_mask0 = cv2.threshold(
        logo_gray, 254, 255, cv2.THRESH_BINARY)
    logo_mask1 = cv2.bitwise_not(logo_mask0)

    img = cv2.imread(oimage, cv2.IMREAD_COLOR)
    lrows, lcols, _ = logo.shape
    irows, icols, _ = img.shape
    x = random.randint(0, irows - lrows)
    y = random.randint(0, icols - irows)

    cimg = img[x: x + lrows, y: y + lcols].copy()
    cimg_and = cv2.bitwise_and(cimg, cimg, mask=logo_mask0)
    logo_and = cv2.bitwise_and(logo, logo, mask=logo_mask1)
    cimg_logo = cv2.add(cimg_and, logo_and)
    cimg_logo = cv2.medianBlur(cimg_logo, 1)
    img[x: x + lrows, y: y + lcols] = cimg_logo[:, :]
    cv2.imwrite(eimage, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--oimages",
                        default="oimages", help="origin images")
    parser.add_argument("-e", "--eimages", default="eimages",
                        help="enhancement images")
    parser.add_argument(
        "-l", "--logo", default="logo/douyin.jpg", help="logo path")
    args = parser.parse_args()
    for filename in os.listdir(args.oimages):
        print("enhance {}".format(filename))
        try:
            combine("{}/{}".format(args.oimages, filename), args.logo,
                    "{}/{}".format(args.eimages, filename), 0.05)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
