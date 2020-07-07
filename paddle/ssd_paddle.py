import os
import argparse
import numpy as np
import cv2
import paddle.fluid as fluid
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

this_dir = os.path.split(os.path.realpath(__file__))[0]
modelname = "mobilenet-ssd"
if modelname == "mobilenet-ssd":
    input_size = 300
else:
    input_size = 160

CLASSES = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax

def preprocess(img):
    img = cv2.resize(img, (input_size, input_size))  
    if modelname == "mobilenet-ssd":
        img = (img - 127.5) * 0.007843
    else:
        mean = (103.94, 116.669, 123.68)
        img = img - mean
    img = img.transpose((2,0,1)).copy()
    img = np.expand_dims(img,axis=0)
    img = img.astype("float32")
    image = fluid.core.PaddleTensor(img)
    return [image]
    
def draw_result(img, out):
    h, w, _ = img.shape
    for dt in out:
        if len(dt) < 5 or dt[1] < 0.5:
            continue
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        xmin=(int)(xmin*w)
        ymin=int(ymin*h)
        xmax=(int)(xmax*w)
        ymax=int(ymax*h)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255))
        if ymin<20:
            ymin=20
        cv2.putText(img,CLASSES[int(dt[0])],(xmin,ymin),3,1,(255,0,0))
    cv2.imshow("img",img)
    cv2.waitKey(1)

def test_image(predictor, imgpath):
    img = cv2.imread(imgpath)
    inputs = preprocess(img)
    outputs = predictor.run(inputs)
    output = outputs[0].as_ndarray()
    draw_result(img, output)
    cv2.waitKey()

def test_camera(predictor):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        inputs = preprocess(img)
        outputs = predictor.run(inputs)
        output = outputs[0].as_ndarray()
        draw_result(img, output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=this_dir+"/"+modelname)
    parser.add_argument("--image", default=this_dir+"/../images/000001.jpg")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #model_file = args.model_dir + "/__model__"
    #params_file = args.model_dir + "/__params__"
    config = AnalysisConfig(args.model_dir)
    config.disable_gpu()
    predictor = create_paddle_predictor(config)
    #test_image(predictor, args.image)
    test_camera(predictor)