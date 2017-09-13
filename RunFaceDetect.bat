@echo off
cd ../../
"Build/x64/Release/ssd_detect" "models/VGGNet/Face2017/SSD_300x300/deploy.prototxt" "models/VGGNet/Face2017/SSD_300x300/VGG_Face2017_SSD_300x300_iter_25359.caffemodel" "list.txt" --confidence_threshold=0.6
pause