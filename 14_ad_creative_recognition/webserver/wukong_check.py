import argparse
import logging
import os
import sys
from keras import optimizers
from os.path import join as pjoin
from wukong.computer_vision.TransferLearning import WuKongVisionModel

# logging
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%d-%H:%M:%S',
            filename=None,
            filemode='w')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wukong checker')
    parser.add_argument('-s', type=int, dest='size', default=224, help='size (optional)')
    parser.add_argument('-w', type=str, dest='weight',
                        default='/home/ec2-user/src/wukong/tmp/douyin_224.combined_model_weightsacc0.83_val_acc0.92.best.hdf5', help='weight (optional)')
    parser.add_argument('-p', type=str, dest='image', help='picture')
    args = parser.parse_args()
    print args
    if args.image is None:
        parser.print_help()
        sys.exit(2)

    try:
        new_model = WuKongVisionModel(args.size, args.size)
        new_model.load_weights(args.weight)
        #ret = new_model.predict('/home/ec2-user/src/leon-hackathon-2018/14_ad_creative_recognition/webserver/upload_image/2035.jpg')
        ret = new_model.predict(args.image)
        if ret[0] > 0.5:
            sys.exit(1) # false
        else:
            sys.exit(0) # true
    except Exception as exc:
        print exc
        sys.exit(3)

