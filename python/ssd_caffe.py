#coding=utf-8
# 极速检测器
# 烟雨@烟雨科技
# V2.0 2020.11.28
import sys
import os
import time
import platform
import argparse
import random
import numpy as np
import cv2
if platform.system()=="Windows":
    caffe_root = 'D:/CNN/ssd'
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

def drawDetection(image, detections, cost = None, usecolor = None):
    if image is None:
        return image
    height, width, _ = image.shape
    for item in detections:
        xmin = int(round(item[3] * width))
        ymin = int(round(item[4] * height))
        xmax = int(round(item[5] * width))
        ymax = int(round(item[6] * height))
        label = CLASSES[int(item[1])-1]
        if usecolor is None:
            cv2.putText(image,label,(xmin,ymin), 3,1,(255,0,0))
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0))
        else:
            import seaborn
            colors = seaborn.color_palette("hls", len(CLASSES))
            color = colors[int(round(item[1]))]
            color = [c *256 for c in color]
            cv2.putText(image,label,(xmin,ymin), 3,1,color)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color)

    if not cost is None:
        cv2.putText(image,cost,(0,40),3,1,(0,0,255))

    return image

class MobileNetSSDDetector:
    def __init__(self,model_def, model_weights):
        if not os.path.exists(model_weights):
            print(model_weights + "does not exist,")
            print("use merge_bn.py to generate it.")
            exit()
        self.net = caffe.Net(model_def,model_weights,caffe.TEST)
        self.net_width = self.net.blobs[self.net.inputs[0]].width
        self.net_height = self.net.blobs[self.net.inputs[0]].height

    def preprocess(self,src):
        img = cv2.resize(src, (self.net_width, self.net_height))
        img = (img - 127.5) * 0.007843
        return img

    def detectAndDraw(self,img):
        start = time.time()
        result = self.detect(img)
        end = time.time()
        cost="%0.2fms" %((end-start)*1000)
        show = drawDetection(img,result, cost)
        return show

    def detect(self,img):
        self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
        #start = time.time()
        detections = self.net.forward()[self.net.outputs[0]][0][0]
        #end = time.time()
        #cost = "det cost: %0.2fms" %((end-start)*1000)
        #print(cost)
        return detections

def testdir(detector, dir = "images"):
    files = os.listdir(dir)
    for file in files:
        imgfile = dir + "/" + file
        img = cv2.imread(imgfile)
        show = detector.detectAndDraw(img)
        cv2.imshow("img",show)
        cv2.waitKey()

def testcamera(detector):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        show = detector.detectAndDraw(img)
        cv2.imshow("img",show)
        cv2.waitKey(1)

def testvideos(detector,dir="videos"):
    files = os.listdir(dir)
    for file in files:
        filepath = dir + "/" + file
        cap = cv2.VideoCapture(filepath)
        while True:
            ret, img = cap.read()
            if not ret:
                break
            show = detector.detectAndDraw(img)
            cv2.imshow("img", show)
            cv2.waitKey(1)

def main(args):
    model_def = rootdir+"/"+args.dataset+'/MobileNetSSD_deploy_depth.prototxt'
    model_weights = rootdir+"/"+args.dataset+'/MobileNetSSD_deploy.caffemodel'
    if not os.path.exists(model_weights):
        print(model_weights+" does not exist,")
        print("use merge_bn.py to generate it.")
        exit()
    if not platform.system() == "Darwin":
        caffe.set_device(0)
        caffe.set_mode_gpu()
    detector = MobileNetSSDDetector(model_def, model_weights)   
    #testcamera(detector)
    #testvideos(detector,args.video_dir)
    testdir(detector)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Face")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)