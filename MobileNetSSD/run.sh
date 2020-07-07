#!/bin/bash
TOOLS=~/CNN/ssd/build/tools/caffe

DATASET=Face
MODEL=MobileNet
INPUT_SHAPE=300x300
PREFIX=${DATASET}_${MODEL}_${INPUT_SHAPE}
#convert data to lmdb
if [ ! -d data/${DATASET}/lmdb ]; then
    #python create_lmdb.py
    python python/convert2lmdb.py
fi

#checkpoint=voc/mobilenet_iter_73000.caffemodel
expdir=output/${PREFIX}

if [ ! -d output ]; then
    mkdir -p output
fi
if [ ! -d ${expdir} ]; then
    mkdir ${expdir}
fi

latest=$(ls -t ${expdir}/*.caffemodel | head -n 1)
snapshot=$(ls -t ${expdir}/*.solverstate | head -n 1)
CMD=""
if test -z $snapshot; then
    if test -z $latest; then
        echo "No checkpoints found!"
    else
        checkpoint=$latest
    fi
    CMD="--weights=$checkpoint"
else
    echo "Resuming from "${latest}
    CMD="--snapshot=${snapshot}"
fi
timestamp=`date "+%Y%m%d_%H%M%S"`
logfile=${expdir}/${timestamp}.log
${TOOLS} train -solver="${DATASET}/solver.prototxt" ${CMD} --gpu 0 2>&1 | tee ${logfile}
# num of objects + 1
# voc: 21
# Car, Face, Hand, Head, Person: 2
NUM_CLASSES=2
if [ ${DATASET} == "voc" ] ; then
    NUM_CLASSES=21
else if [ ${DATASET} == "Mask" ] ; then
    NUM_CLASSES=3
    fi
fi
rm -rf example
./gen_model.sh ${NUM_CLASSES}

if [ ${DATASET} == "voc" ] ; then
    deploytxt="voc/deploy.prototxt"
else
    deploytxt="example/MobileNetSSD_deploy.prototxt"
fi
latest=$(ls -t ${expdir}/*.caffemodel | head -n 1)
python python/merge_bn.py --model=${deploytxt} --weight=${latest} --outmodel="${expdir}/deploy.prototxt" --outweight="${DATASET}/MobileNetSSD_deploy.caffemodel"
echo "merged "$deploytxt ${latest}
#demo
python python/demo.py --modeldir=${DATASET}