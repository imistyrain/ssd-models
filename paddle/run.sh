#!/bin/bash

#convert caffe model to paddle
if [ ! -d mobilenet-ssd ]; then
    x2paddle --framework=caffe --prototxt=../voc/MobileNetSSD_deploy.prototxt --weight=../voc/MobileNetSSD_deploy.caffemodel --save_dir=./ --params_merge
    mv inference_model mobilenet-ssd
fi

python3 ssd_paddle.py