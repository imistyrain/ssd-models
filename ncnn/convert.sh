#!/bin/bash
MODEL=mask #$(basename `pwd`)
echo "converting "${MODEL}
NCNN_DIR=~/CNN/ncnn
TOOLS_DR=${NCNN_DIR}/build/tools
caffe2ncnn=${TOOLS_DR}/caffe/caffe2ncnn
ncnnoptimize=${TOOLS_DR}/ncnnoptimize
ncnn2table=${TOOLS_DR}/quantize/ncnn2table
ncnn2int8=${TOOLS_DR}/quantize/ncnn2int8

#convert to ncnn
if [ ! -f ${MODEL}.param ] ; then
    $caffe2ncnn ${MODEL}.prototxt ${MODEL}.caffemodel ${MODEL}-bn.param ${MODEL}-bn.bin
fi
#optimize
if [ ! -f  ${MODEL}.param ]; then
    $ncnnoptimize ${MODEL}-bn.param ${MODEL}-bn.bin ${MODEL}.param ${MODEL}.bin 0
fi
#get table
if [ ! -f ${MODEL}.table ] ; then
    $ncnn2table --param=${MODEL}.param --bin=${MODEL}.bin --images=../images --output=${MODEL}.table --mean=127.5,127.5,127.5 --norm=0.007843,0.007843,0.007843 --size=300,300 --thread=4 --swapRB=0
fi
#convert to int8
if [ ! -f ${MODEL}-int8.param ];then
    $ncnn2int8 ${MODEL}.param ${MODEL}.bin ${MODEL}-int8.param ${MODEL}-int8.bin ${MODEL}.table
fi