#!/bin/sh
cd ..
TOOLS=~/Detection/ssd/build/tools/caffe
DATASET=yufacenet
PREFIX=shufflenet
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