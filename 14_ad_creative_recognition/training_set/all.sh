#!/usr/bin/env bash

mkdir oimage
mkdir eimage

python spiders/photo.py
python image_enhancement/image_enhancement.py  
