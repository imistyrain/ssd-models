import numpy as np  
import sys,os  
import cv2
import caffe

dataset="voc"
net_file= dataset+'/MobileNetSSD_deploy.prototxt'  
caffe_model=dataset+'/MobileNetSSD.caffemodel'
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')
input_size=(300,300)

def preprocess(img):
    img = cv2.resize(img, input_size)
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(img):
    data = preprocess(img)
    net.blobs['data'].data[...] = data.transpose(2, 0, 1)
    out = net.forward()  
    box, conf, cls = postprocess(img, out)
    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       cv2.rectangle(img, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(img, title, p3, 1, 1, (0, 255, 0), 1)
    return img

def test_camera():   
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = detect(img)
        cv2.imshow("img", img)
        cv2.waitKey(1)

def testdir(dir="images"):
    files=os.listdir(dir)
    for file in files:
        imgfile=dir+"/"+file
        img = cv2.imread(imgfile)
        img = detect(img)
        cv2.imshow("img", img)
        cv2.waitKey()
        
if __name__=="__main__":
    testdir()
    #test_camera()