@echo off
set NCNN_DIR=D:/CNN/ncnn
"%NCNN_DIR%/build/tools/caffe/Release/caffe2ncnn" MobileNetSSD_deploy_ncnn.prototxt MobileNetSSD_deploy.caffemodel ncnn_face.param ncnn_face.bin
pause