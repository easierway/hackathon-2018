import cv2
import json
import logging
import os
import subprocess
import sys
import urllib2
import uuid
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from mimetypes import guess_extension
from os.path import basename, join as pjoin, dirname, realpath, splitext
from werkzeug.utils import secure_filename

import colorlog
from keras import optimizers
from wukong.computer_vision.TransferLearning import WuKongVisionModel


# constants
dir_path = dirname(realpath(__file__))
UPLOAD_IMAGE_FOLDER = pjoin(dir_path, 'upload_image')
UPLOAD_VIDEO_FOLDER = pjoin(dir_path, 'upload_video')
TRAIN_IMAGE_FOLDER = pjoin(dir_path, 'train_image')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'jpe'}

# logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d-%H:%M:%S',
                    filename=None,
                    filemode='w')

addresses = [
    "http://127.0.0.1:5700",
    "http://127.0.0.1:5672",
    "http://127.0.0.1:5448",
    "http://127.0.0.1:5300",
    "http://127.0.0.1:5224",
]


def predict(image):
    for address in addresses:
        try:
            response = urllib2.urlopen("{}?image={}".format(address, image))
            obj = json.loads(response.read())
            logging.info("predict [{}] [{}] [{}] [{}]".format(
                address, image, obj["value"], obj))
            if "status" in obj and obj["status"] == 0 and obj["recognition"] == 1:
                return True
        except Exception as e:
            logging.exception(e)
    return False


def download(url, output="./upload_image"):
    try:
        dlfile = urllib2.urlopen(url)
        extension = guess_extension(dlfile.info()['Content-Type'])
        tempname = str(uuid.uuid4()) + extension
        tempfile = pjoin(output, tempname)

        with open(tempfile, 'wb') as output:
            output.write(dlfile.read())
        logging.info("download %r to %r", url, tempfile)
        return tempfile
    except Exception as exc:
        logging.error("download exception %s", exc)
        return None


def allowed_file(filename):
    logging.debug("check allowed file %r", filename)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    _, extension = splitext(filename)
    return extension in (".mp4", ".MP4", ".avi", ".AVI", ".MOV", ".mov")


def pick_frame(inPath, outPath, num=5):
    logging.debug("pick frame:%r, out path:%r", inPath, outPath)
    frame_list = []
    videoName = basename(inPath)

    videoCapture = cv2.VideoCapture()
    videoCapture.open(inPath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    if -1 == num or num > frames:
        num = frames
    for i in range(int(num)):
        ret, frame = videoCapture.read()
        if ret is True:
            frame_save_path = "%s/%s-frame-%d.jpg" % (outPath, videoName, i)
            cv2.imwrite(frame_save_path, frame)
            frame_list.append(frame_save_path)
            logging.info("pick frame save %r", frame_save_path)

    if len(frame_list) > 1:
        frame_list.pop(0)

    return frame_list


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognition', methods=['POST'])
def recognition():
    imageurl = request.form["imageUrl"]
    logging.debug("imageurl %r", imageurl)
    if 'file' not in request.files and not imageurl:
        flash('No file part')
        return json.dumps({"status": 1, "message": "you should specific at least one of image or url"})

    try:
        predictResult = False
        if imageurl:
            dlfile = download(url=imageurl, output=UPLOAD_IMAGE_FOLDER)
            if dlfile is None or not allowed_file(basename(dlfile)):
                return json.dumps({"status": 2, "message": "not allow file format"})
            predictResult = predict(dlfile)
        else:
            file = request.files['file']
            if file.filename == '':
                return json.dumps({"status": 3, "message": "no selected file"})
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # video
                if is_video(filename):
                    upload_save_path = pjoin(UPLOAD_VIDEO_FOLDER, filename)
                    file.save(upload_save_path)
                    logging.debug("video save %r", upload_save_path)
                    frames = pick_frame(upload_save_path, UPLOAD_IMAGE_FOLDER)
                    for f in frames:
                        predictResult = predict(f)
                        if predictResult is True:
                            break
                else:
                    upload_save_path = pjoin(UPLOAD_IMAGE_FOLDER, filename)
                    file.save(upload_save_path)
                    logging.debug("image save %r", upload_save_path)
                    predictResult = predict(upload_save_path)
    except Exception as e:
        logging.exception(e)
        return json.dumps({"status": 4, "message": "server interval error"})

    return json.dumps({
        "status": 0,
        "recognition": predictResult is True
    })


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
