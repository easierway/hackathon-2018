import cv2
import logging
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from os.path import basename, join as pjoin, dirname, realpath, splitext
from werkzeug.utils import secure_filename

# constants
dir_path = dirname(realpath(__file__))
UPLOAD_IMAGE_FOLDER = pjoin(dir_path, 'upload_image')
UPLOAD_VIDEO_FOLDER = pjoin(dir_path, 'upload_video')
TRAIN_IMAGE_FOLDER = pjoin(dir_path, 'train_image')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# logging
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%d-%H:%M:%S',
            filename=None,
            filemode='w')

# flask
app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pick_frame(inPath, outPath, num=1):
    logging.debug("pick frame:%r, out path:%r", inPath, outPath)
    frame_list = []
    videoName = basename(inPath)
    videoPath = dirname(inPath)

    videoCapture = cv2.VideoCapture()
    videoCapture.open(inPath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    if -1 == num or  num > frames:
        num = frames
    for i in range(int(num)):
        ret,frame = videoCapture.read()
        if ret is True:
            frame_save_path = "%s/%s-frame-%d.jpg" % (outPath, videoName, i)
            cv2.imwrite(frame_save_path, frame)
            frame_list.append(frame_save_path)
            logging.info("pick frame save %r", frame_save_path)

    return frame_list


def is_video(filename):
    _, extension = splitext(filename)
    return extension in (".mp4", ".avi")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # video
            if is_video(filename):
                upload_save_path = pjoin(UPLOAD_VIDEO_FOLDER, filename)
                file.save(pjoin(upload_save_path))
                logging.debug("video save %r", upload_save_path)
                frames = pick_frame(upload_save_path, UPLOAD_IMAGE_FOLDER)
                if len(frames) > 0:
                    # TODO: return predict picture
                    filename = basename(frames[0])
            else:
                upload_save_path = pjoin(UPLOAD_IMAGE_FOLDER, filename)
                file.save(pjoin(upload_save_path))
                logging.debug("image save %r", upload_save_path)

            # return redirect(url_for('uploaded_file', filename=filename))
            return render_template('result.html',
                                   uploaded=url_for('uploaded_file', filename=filename),
                                   result=url_for('static', filename='true.jpg'))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/upload_image/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_IMAGE_FOLDER, filename)

