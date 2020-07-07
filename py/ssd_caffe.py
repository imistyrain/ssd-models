#coding=utf-8
#基于MobileNet-SSD的检测
#烟雨@烟雨科技
#V1.0.0 2018.03.18
import sys,os,time,platform,argparse,random
import numpy as np
import seaborn as sns
import cv2
if platform.system()=="Windows":
    caffe_root = 'D:/CNN/ssd'
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

rootdir=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

WINDOWSNAME="ssdcaffe"

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def drawDetectionResult(image,result,colors=None,cost=None):
    if image is None:
        return image
    height,width,c=image.shape
    show=image.copy()
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))
        
        if colors is None:
            cv2.putText(show,item[6],(xmin,ymin), 3,1,(0,255,0))
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),(0,255,0))
        else:
            color=colors[int(round(item[4]))]
            color=[c *256 for c in color]
            cv2.putText(show,item[6],(xmin,ymin), 3,1,color)
            cv2.rectangle(show,(xmin,ymin),(xmax,ymax),color)

    if not cost is None:
        cv2.putText(show,cost,(0,40),3,1,(0,0,255))

    return show

class MobileNetSSDDetection:
    def __init__(self,model_def, model_weights,image_resize, labelmap_file):
        self.image_resize = image_resize
        if not os.path.exists(model_weights):
            print("MobileNetSSD_deploy.caffemodel does not exist,")
            print("use merge_bn.py to generate it.")
            exit()
        self.net = caffe.Net(model_def,model_weights,caffe.TEST)
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)
        self.colors =sns.color_palette("hls", len(self.labelmap.item))

    def preprocess(self,src):
        img = cv2.resize(src, (self.image_resize,self.image_resize))
        img = (img - 127.5)* 0.007843
        return img

    def detectAndDraw(self,img):
        start=time.time()
        result=self.detect(img)
        end=time.time()
        cost="%0.2fms" %((end-start)*1000)
        show=drawDetectionResult(img,result,self.colors,cost)
        return show

    def detect(self,img, conf_thresh=0.5, topn=10):
        self.net.blobs['data'].data[...] = self.preprocess(img).transpose((2, 0, 1)) 
        #start=time.time()
        detections = self.net.forward()['detection_out']
        #end=time.time()
        #cost="%0.2fms" %((end-start)*1000)
        #print(cost)
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in range(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i]
            ymin = top_ymin[i]
            xmax = top_xmax[i]
            ymax = top_ymax[i]
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def testdir(detection,dir):
    files=os.listdir(dir)
    for file in files:
        imgfile=dir+"/"+file
        img = cv2.imread(imgfile)
        show=detection.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey()

def testcamera(detection,index=0):
    cap=cv2.VideoCapture(index)
    while True:
        ret,img=cap.read()
        if not ret:
            break
        show=detection.detectAndDraw(img)
        cv2.imshow(WINDOWSNAME,show)
        cv2.waitKey(1)

def testvideos(detection,dir):
    files=os.listdir(dir)
    for file in files:
        filepath=dir+"/"+file
        cap=cv2.VideoCapture(filepath)
        while True:
            ret,img=cap.read()
            if not ret:
                break
            show=detection.detectAndDraw(img)
            cv2.imshow(WINDOWSNAME,show)
            cv2.waitKey(1)

def main(args):
    '''main '''
    model_def= rootdir+"/"+args.dataset+'/MobileNetSSD_deploy_depth.prototxt'
    model_weights= rootdir+"/"+args.dataset+'/MobileNetSSD_deploy.caffemodel'
    if not os.path.exists(model_weights):
        print("MobileNetSSD_deploy.caffemodel does not exist,")
        print("use merge_bn.py to generate it.")
        exit()
    labelmap_file=rootdir+"/"+args.dataset+"/labelmap.prototxt"
    #caffe.set_device(args.gpu_id)
    #caffe.set_mode_gpu()
    detection = MobileNetSSDDetection(model_def, model_weights,
                            args.image_resize, labelmap_file)   
    testcamera(detection)
    #testvideos(detection,args.video_dir)

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="voc", help='dataset')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--cameraindex', default=0, type=int)
    parser.add_argument('--video_dir', default=rootdir+"/videos")
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())