import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf.text_format import Merge
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default='Face/MobileNetSSD_train.prototxt')
    parser.add_argument('--target',default='Face/MobileNetSSD_train_depth.prototxt')
    args = parser.parse_args()
    net = caffe_pb2.NetParameter()
    Merge(open(args.model, 'r').read(), net)
    for layer in net.layer:
        if layer.type == "Convolution":
            if layer.convolution_param.group !=1:
                layer.type = "DepthwiseConvolution"
    with open(args.target, 'w') as f:
        f.write(str(net))