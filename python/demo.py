import platform
import os
import sys
if platform.system() == "Windows":
    caffe_root = "D:/CNN/ssd"#2
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root+'/python')
import caffe
import cv2
import time
import argparse
import numpy as np

WINDOWSNAME="ssd-models"

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
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(255,0,0))
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
        start=time.time()
        result=self.detect(img)
        end=time.time()
        cost="%0.2fms" %((end-start)*1000)
        show=drawDetection(img,result,cost=cost)
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
        paded = cv2.copyMakeBorder(src,self.top,self.bottom,self.left,self.right,0)
        img = cv2.resize(paded, (self.width,self.height))
        # img = img - [104, 117, 123]
        img = (img - 127.5)* 0.007843
        return img.transpose(2,0,1)

    def showpriors(self):
        blobs = self.net.blobs
        blobs = [b for b in blobs if b.endswith("mbox_priorbox")]
        input = self.net.blobs[self.net.inputs[0]]
        self.net.forward()
        for b in blobs:
            priorboxs = self.net.blobs[b].data[0][0].reshape(-1,4)
            img = np.zeros((input.height,input.width,3),dtype=np.uint8)
            for prior in priorboxs:
                x1 = int(prior[0]*img.shape[1])
                y1 = int(prior[1]*img.shape[0])
                x2 = int(prior[2]*img.shape[1])
                y2 = int(prior[3]*img.shape[0])
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))
            cv2.imshow("img",img)
            cv2.waitKey()

def testdir(detector,dir="images"):
    files=os.listdir(dir)
    for file in files:
        imgfile=dir+"/"+file
        img = cv2.imread(imgfile)
        show=detector.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey()

def testcamera(detector,index=0):
    cap=cv2.VideoCapture(index)
    while True:
        ret,img=cap.read()
        if not ret:
            break
        show=detector.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey(1)

def main(args):
    if not platform.system()=="Darwin":
        caffe.set_device(0)
        caffe.set_mode_gpu()
    model_def = args.modeldir + '/'+args.modeldef
    model_weights = args.modeldir +  '/'+args.weight
    detector = Detection(model_def, model_weights)
    #detector.showpriors()
    testdir(detector)
    #testcamera(detector)
    
def get_args():
    dataset="Face"
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default=dataset, help='modeldir')
    parser.add_argument('--modeldef', default="MobileNetSSD_deploy_depth.prototxt")
    parser.add_argument('--weight', default="MobileNetSSD_deploy.caffemodel")
    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    main(args)