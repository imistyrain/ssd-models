import os
import cv2
import numpy as np
import sys
caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root+'/python')
import caffe
from tqdm import tqdm

CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 
'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 
'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

# color index please refer to https://zhuanlan.zhihu.com/p/102303256
colors = [[0,0,0], [128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],
        [0,0,128],[128,128,128], [64,0,0],[192,0,0],[64,128,0],
        [192,128,0], [64,0,128], [192,0,128],  [64,128,128], [192,128,128],
        [0,64,0], [128,64,0], [0,192,0], [128,192,0],[0,64,128]]

outputdir="output/preproess"

def showpreprocess(blobs,i,show=False):
    data = np.array(blobs['data'].data)
    label = np.array(blobs['label'].data)
    img = data[0].transpose(1,2,0).copy()
    objs = label[0][0]
    height, width,_ = img.shape
    for obj in objs:
        x = int(obj[3]*width)
        y = int(obj[4]*height)
        x2 = int(obj[5]*width)
        y2 = int(obj[6]*height)
        cls = int(obj[1])
        cv2.rectangle(img,(x,y),(x2,y2),colors[cls])
        cv2.putText(img,CLASSES[cls],(x,y),1,1,colors[cls])

    if show:
        cv2.imshow("img",img)
        cv2.waitKey()
    cv2.imwrite(outputdir+"/"+str(i)+".jpg",img)

def main(model="voc/MobileNetSSD_preprocess.prototxt",show=False):
    net = caffe.Net(model, caffe.TRAIN)
    for i in tqdm(range(20)):
        blobs = net.forward()
        showpreprocess(blobs,i)

if __name__=="__main__":
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    main()