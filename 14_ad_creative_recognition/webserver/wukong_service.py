#!/usr/bin/python
import colorlog
import argparse
import json
import logging
import os
import sys
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from keras import optimizers
from os.path import join as pjoin
from urlparse import urlparse, parse_qs
from wukong.computer_vision.TransferLearning import WuKongVisionModel

# global wukong model
wukong = None

# logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt='%d-%H:%M:%S'))
logger = colorlog.getLogger('root')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


#This class will handles any incoming request from
#the browser 
class myHandler(BaseHTTPRequestHandler):

    #Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        query_str = urlparse(self.path).query
        query = dict(qc.split("=") for qc in query_str.split("&"))
        #print query
        predict_list =  wukong.predict(query["image"])
        predict = False
        if predict_list[0] < 0.5:
            predict = True
            
        # Send the html message
        self.wfile.write(json.dumps({"recognition": predict, "value":str(predict_list[0]), "status":0, "message":""}))

        return

    def init_model(self, size, weight):
        self.model = WuKongVisionModel(size, size)
        self.model.load_weights(weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wukong checker service')
    parser.add_argument('-s', type=int, dest='size', default=672, help='size')
    parser.add_argument('-w', type=str, dest='weight',
                        default='/home/ec2-user/src/wukong/tmp/douyin_672.combined_model_weightsacc0.921_val_acc0.993.best.hdf5', help='weight')
    parser.add_argument('-p', type=int, dest='port', default=50000, help='port')
    args = parser.parse_args()
    if args.size is None or args.weight is None or args.port is None:
        parser.print_help()
        sys.exit(2)

    try:

        #Create a web server and define the handler to manage the
        #incoming request
        server = HTTPServer(('', args.port), myHandler)
        print 'Started httpserver on port ' , args.port

        global wukong
        wukong = WuKongVisionModel(args.size, args.size)
        wukong.load_weights(args.weight)

        #Wait forever for incoming htto requests
        server.serve_forever()

    except KeyboardInterrupt:
        print '^C received, shutting down the web server'


