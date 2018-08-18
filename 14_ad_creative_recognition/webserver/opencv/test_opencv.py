#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import os

def pickFrame(inPath, outPath, num):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(inPath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    #fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
    #print("fps=",fps,"frames=",frames)
    if -1 == num or  num > frames:
        num = frames 
    for i in range(int(num)):
        ret,frame = videoCapture.read()
        if ret is True:
            cv2.imwrite("%s-frame-%d.jpg" % (inPath, i), frame)

if __name__ == "__main__":
    inPath = "douyin_video1.mp4"
    outPath = ""
    num = 3
    pickFrame(inPath, outPath, num)

