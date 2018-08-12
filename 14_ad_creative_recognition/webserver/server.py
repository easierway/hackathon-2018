import cv2
import logging
import os
import subprocess
import sys
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from os.path import basename, join as pjoin, dirname, realpath, splitext
from werkzeug.utils import secure_filename

# use system wukong
# sys.path.append("./wukong")
# from wukong.computer_vision.TransferLearning import WuKongVisionModel


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


# wukong
class WuKong:
    def __init__(self, size, weight):
        self.wukong_dir = ""
        self.model = WuKongVisionModel(size, size)
        self.model.load_weights(weight)


    def train(self):
        '''train a model with the default configuration'''
        self.model.train_for_new_task(self.work_dir, self.task_name, self.train_data_dir, self.test_data_dir)


    def predict(self, filepath):
        '''predict by the trained model'''
        ret = self.model.predict(filepath)
        return ret


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    _, extension = splitext(filename)
    return extension in (".mp4", ".avi")


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
            result_image = 'false.jpg'
            predict = -1

            # video
            if is_video(filename):
                upload_save_path = pjoin(UPLOAD_VIDEO_FOLDER, filename)
                file.save(pjoin(upload_save_path))
                logging.debug("video save %r", upload_save_path)
                frames = pick_frame(upload_save_path, UPLOAD_IMAGE_FOLDER)
                for f in frames:
                    filename = basename(f)
                    predict = subprocess.call(['python', 'wukong_check.py', '-p', f])
                    logging.info("predict frame %r = %r", predict, f)
                    if predict == 0:
                        break
            else:
                upload_save_path = pjoin(UPLOAD_IMAGE_FOLDER, filename)
                file.save(upload_save_path)
                logging.debug("image save %r", upload_save_path)
                # predict = app.wukong.predict(upload_save_path)
                predict = subprocess.call(['python', 'wukong_check.py', '-p', upload_save_path])
                logging.info("predict image %r = %r", predict, upload_save_path)

            if predict == 0:
                result_image = 'true.jpg'

            # return redirect(url_for('uploaded_file', filename=filename))
            return render_template('result.html',
                                   uploaded=url_for('uploaded_file', filename=filename),
                                   result=url_for('static', filename=result_image))

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


if __name__ == '__main__':
    #size = 224
    #weight = '/home/ec2-user/src/wukong/tmp/douyin_448.combined_model_weightsacc0.90_val_acc0.99.best.hdf5'
    #wukong = WuKong(size, weight)
    #app.wukong = wukong
    app.run(host='0.0.0.0',port=5000)

