@echo off
cd ../../
"build/examples/ssd/ssd_detect" "data/ssd-face/cpp/models/face_deploy.prototxt" "data/ssd-face/cpp/models/VGG_Face2017_SSD_300x300_iter_120000.caffemodel" "list.txt" --confidence_threshold=0.6
pause