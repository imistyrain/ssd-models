#coding=utf-8
import platform
import os
import sys

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# 按数字依次修改
# 1. 修改路径为自己电脑ssd编译路径
if platform.system() == "Windows":
    caffe_root = "D:/CNN/ssd"
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe
import cv2
import time
import argparse
import numpy as np

def get_latest_model(dir="./"):
    files = os.listdir(dir)
    max_iter = 0
    latest = None
    for file in files:
        if file.endswith("caffemodel"):
            iter  = int(file.split(".")[0].split("_")[-1])
            if iter > max_iter:
                max_iter = iter
                latest = file
    return dir + "/" + latest

def drawDetection(image,detections,colors=None,cost=None):
    if image is None:
        return image
    for item in detections:
        xmin = item[3]
        ymin = item[4]
        xmax = item[5]
        ymax = item[6]
        label = str(int(item[1]))
        if colors is None:
            cv2.putText(image,label,(xmin,ymin), 3,1,(255,0,255))
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0),10)
        else:
            color=colors[int(round(item[1]))]
            color=[c *256 for c in color]
            cv2.putText(image,label,(xmin,ymin), 3,1,color)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color)

    if not cost is None:
        cv2.putText(image,cost,(0,40),3,1,(0,0,255))

    return image

class Detection:
    def __init__(self,model_def, model_weights):
        if not os.path.exists(model_weights):
            print(model_weights + " does not exist,")
            exit()
        self.net = caffe.Net(model_def,model_weights,caffe.TEST)
        self.height = self.net.blobs['data'].shape[2]
        self.width = self.net.blobs['data'].shape[3]
        self.ratio = self.width * 1.0 / self.height
        self.scale = 1.0
        self.top = 0
        self.bottom = 0
        self.left = 0
        self.right = 0

    def detectAndDraw(self,img):
        start = time.time()
        result = self.detect(img)
        end = time.time()
        cost = "%0.2fms" %((end-start)*1000)
        show = drawDetection(img, result, cost=cost)
        return show
    
    def detect(self,img,threshold=0.28):
        self.net.blobs[self.net.inputs[0]].data[...] = self.preprocess(img)
        detections = self.net.forward()[self.net.outputs[0]][0][0]
        for detection in detections:
            if detection[2] < threshold:
                continue
            detection[3] = detection[3]*self.width*self.scale - self.left
            detection[4] = detection[4]*self.height*self.scale - self.top
            detection[5] = detection[5]*self.width*self.scale - self.left
            detection[6] = detection[6]*self.height*self.scale - self.top
        return detections
    
    def preprocess(self,src):
        h, w, _ = src.shape
        self.left = 0
        self.right = 0
        self.bottom = 0
        self.top = 0
        if w * 1.0 / h > self.ratio:
            self.bottom = int((w /self.ratio - h ) /2)
            self.top = int(w / self.ratio) - h - self.bottom
            self.scale = w * 1.0 / self.width
        else:
            self.left = int((h * self.ratio- w )/2)
            self.right = int(h * self.ratio) - w - self.left
            self.scale = h * 1.0 / self.height
        paded = cv2.copyMakeBorder(src, self.top, self.bottom, self.left, self.right, 0)
        img = cv2.resize(paded, (self.width, self.height))
        # for caffe ssd preprocess
        #img = img - [104, 117, 123]
        # for mobilenet-ssd preprocess
        img = (img - 127.5)* 0.007843
        return img.transpose(2,0,1)

def testdir(detector, dir = "images"):
    files = os.listdir(dir)
    for file in files:
        imgfile = dir + "/" + file
        img = cv2.imread(imgfile)
        show = detector.detectAndDraw(img)
        cv2.imshow("img",show)
        cv2.waitKey()

def testcamera(detector, index = 0):
    cap = cv2.VideoCapture(index)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        show = detector.detectAndDraw(img)
        cv2.imshow("img",show)
        cv2.waitKey(1)

def main(args):
    if not platform.system() == "Darwin":
        caffe.set_device(0)
        caffe.set_mode_gpu()
    if args.weight is None:
        args.weight = get_latest_model(rootdir)
    detector = Detection(rootdir+"/"+args.model, args.weight)
    #detector.showpriors()
    #testdir(detector)
    testcamera(detector)
    
def get_args():
    parser = argparse.ArgumentParser()
    # 2. 修改模型路径为release里下载到本地的模型路径
    parser.add_argument('--model', default = "deploy.prototxt")
    parser.add_argument('--weight', default = None)
    return parser.parse_args()
    
if __name__ == '__main__':
    args = get_args()
    main(args)
