# -*- coding: utf-8 -*
import numpy as np  
import sys,os,cv2,platform
if platform.system()=="Windows":
    caffe_root="D:/CNN/ssd"
else:
    caffe_root="/home/yanyu/Detection/ssd"
sys.path.insert(0,caffe_root+'/python')
sys.path.insert(0, caffe_root + '/python')  
import caffe  
import time
net_file= 'cpp/models/faceboxes_deploy.prototxt'  
caffe_model='cpp/models/FaceBoxes_1024x1024.caffemodel'  
test_dir = "images"

if not os.path.exists(caffe_model):
    print("FaceBoxes_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
caffe.set_mode_gpu()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'face')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel

def preprocess(src):
    img = cv2.resize(src, (1024,1024))
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

def testimg(imgfile):
    frame = cv2.imread(imgfile)
    #res =frame
    res = cv2.resize(frame, (1024, 1024), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
    transformed_image = transformer.preprocess('data', res)
    # print img
    net.blobs['data'].data[...] = transformed_image
    time_start=time.time()
    out = net.forward()  
    time_end=time.time()
    print(time_end-time_start)
    box, conf, cls = postprocess(frame, out)
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(frame, p1, p2, (0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("faceboxes", frame)
    cv2.waitKey()
def detect(cameraindex=0):
    videoCapture = cv2.VideoCapture(cameraindex)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH,640)  
  
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)  

    success, frame = videoCapture.read()
    while success :
      success, frame = videoCapture.read()

      res = cv2.resize(frame, (1024, 1024), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)
      #res =frame
      transformed_image = transformer.preprocess('data', frame)
      # print img
      net.blobs['data'].data[...] = transformed_image


      time_start=time.time()
      out = net.forward()  
      time_end=time.time()
      print(time_end-time_start)

      box, conf, cls = postprocess(frame, out)

      for i in range(len(box)):
         p1 = (box[i][0], box[i][1])
         p2 = (box[i][2], box[i][3])
         cv2.rectangle(frame, p1, p2, (0,255,0))
         p3 = (max(p1[0], 15), max(p1[1], 15))
         title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
         cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
      cv2.imshow("SSD", frame)
     

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      #Exit if ESC pressed
      #if k == 27 : return False
    return True

if __name__=="__main__":
    testimg("cpp/000001.jpg")