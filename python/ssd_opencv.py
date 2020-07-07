import os
import numpy as np
import argparse
import time
import cv2

rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Face")
    parser.add_argument("--input_shape",default = [300, 300])
    parser.add_argument("--num_classes", default = 2)
    parser.add_argument("--thr", default = 0.4)
    return parser.parse_args()

def detect(net, img, input_shape = (300,300), mean =(127.5, 127.5, 127.5), scale = 0.007843, swapRB = False, threshold = 0.4):
    blob = cv2.dnn.blobFromImage(img, scale, input_shape, mean, swapRB)
    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    end = time.time()
    print((end - start) * 1000)
    detections = detections.reshape(-1, 7)
    h, w, _ = img.shape
    for detection in detections:
        if detection[2] < threshold:
            continue
        label = int(detection[1])
        x = int(detection[3] * w)
        y = int(detection[4] * h)
        x2 = int(detection[5] * w)
        y2 = int(detection[6] * h)
        cv2.putText(img,str(label),(x,y),1,1,(255,0,0))
        cv2.rectangle(img, (x,y), (x2,y2), (255,0,0))
    return img

def test_camera(net, args):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        show = detect(net, img)
        cv2.imshow("img", show)
        cv2.waitKey(1)

def test_dir(net,dir="images"):
    files = os.listdir(dir)
    for file in files:
        imgfile = dir + "/"+file
        img = cv2.imread(imgfile)
        show = detect(net, img)
        cv2.imshow("img",show)
        cv2.waitKey()

if __name__ == "__main__":
    args = get_args()
    model_def = rootdir+"/"+args.dataset+'/MobileNetSSD_deploy.prototxt'
    model_weights = rootdir+"/"+args.dataset+'/MobileNetSSD_deploy.caffemodel'
    net = cv2.dnn.readNetFromCaffe(model_def, model_weights)
    #test_camera(net,args)
    test_dir(net)