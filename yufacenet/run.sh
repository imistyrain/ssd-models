#!/bin/bash
cd ..
TOOLS=~/CNN/ssd/build/tools/caffe

# voc 21
# Face, Car, Person, Head 2
DATASET=yufacenet

# NUM_CLASSES=2
# rm -rf example
# ./gen_model.sh ${NUM_CLASSES}

PREFIX=${DATASET}
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

latest=$(ls -t ${expdir}/*.caffemodel | head -n 1)
echo "Testing "${latest}
#${TOOLS} train -solver="${DATASET}/solver.prototxt" --weights=$latest

#python py/merge_bn.py --model="${DATASET}/deploy_nobn.prototxt" --weight=${latest} --outmodel="${expdir}/no_bn.prototxt" --outweight="${DATASET}/MobileNetSSD.caffemodel"
echo "merged "${latest}
#demo
#python demo.py --modeldir=${DATASET}
