#!/bin/bash
cd ../
TOOLS=~/Face/FaceBoxes/build/tools/caffe

DATASET=faceboxes
MODEL=""
INPUT_SHAPE=300x300
PREFIX=${DATASET}_${MODEL}_${INPUT_SHAPE}
#convert data to lmdb
#softlink data

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

${TOOLS} train -solver="${DATASET}/solver_train.prototxt" ${CMD} --gpu 0 \
2>&1 | tee ${expdir}/train.log

latest=$(ls -t ${expdir}/*.caffemodel | head -n 1)
echo "Testing "${latest}
# test
#${TOOLS} train -solver="${DATASET}/solver_test.prototxt" --weights=$latest
#demo
#python demo.py --modeldir=${expdir} --modeldef=deploy.prototxt --weight=${latest}
