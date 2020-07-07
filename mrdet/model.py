import caffe
from caffe import layers as L
from caffe import params as P

#mobile net
def ConvDwPw(net, from_layer,out_layer, num_output_dw,num_output_pw, stride,lr_mult,use_global_stats = True):
    bn_lr_mult=0
    name = out_layer
    kwargs = {
      'param': [dict(lr_mult=lr_mult, decay_mult=1)],
      'weight_filler': dict(type='gaussian', std=0.01),
      'bias_term': False,
      'engine': 1,
    }
    bn_kwargs = {
    'param': [
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0)],
    'use_global_stats': use_global_stats,
    }
    sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
    }
    net[name+'/dw'] = L.Convolution(net[from_layer],num_output = num_output_dw,kernel_size = 3,pad = 1,group = num_output_dw, stride = stride,**kwargs)
    net[name+'/dw/bn'] = L.BatchNorm(net[name+'/dw'],in_place=True,**bn_kwargs)
    net[name+'/dw/scale']=L.Scale(net[name+'/dw'],in_place=True,**sb_kwargs)
    net[name+'/dw/relu'] = L.ReLU(net[name+'/dw'],in_place=True)
    kwargs = {
      'param': [dict(lr_mult=lr_mult, decay_mult=1)],
      'weight_filler': dict(type='gaussian', std=0.01),
      'bias_term': False,
    }
    net[name] = L.Convolution(net[name+'/dw'],num_output = num_output_pw,kernel_size = 1,pad = 0,stride = 1,**kwargs)
    net[name+'/bn'] = L.BatchNorm(net[name],in_place=True,**bn_kwargs)
    net[name+'/scale']=L.Scale(net[name],in_place=True,**sb_kwargs)
    net[name+'/relu'] = L.ReLU(net[name],in_place=True)
def MobileNetBody(net,from_layer="data",lr_mult = 1,use_global_stats=True):
    bn_lr_mult=0
    kwargs = {
      'param': [dict(lr_mult=lr_mult, decay_mult=1)],
      'weight_filler': dict(type='gaussian', std=0.01),
      'bias_term': False,
    }
    bn_kwargs = {
      'param': [
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0),
        dict(lr_mult=0, decay_mult=0)],
      'use_global_stats': use_global_stats,
    }
    sb_kwargs = {
          'bias_term': True,
          'param': [
              dict(lr_mult=bn_lr_mult, decay_mult=0),
              dict(lr_mult=bn_lr_mult, decay_mult=0)],
          'filler': dict(type='constant', value=1.0),
          'bias_filler': dict(type='constant', value=0.0),
    }
    net['conv0'] = L.Convolution(net[from_layer],num_output = 32,kernel_size = 3,stride = 2,pad = 1,**kwargs)
    net['conv0/bn'] = L.BatchNorm(net['conv0'],in_place=True,**bn_kwargs)
    net['conv0/scale'] = L.Scale(net['conv0'],in_place = True,**sb_kwargs)
    net['conv0/relu'] = L.ReLU(net['conv0'],in_place = True)
    ConvDwPw(net,from_layer='conv0',lr_mult=lr_mult,out_layer='conv1',num_output_dw=32,num_output_pw=64,stride = 1)
    ConvDwPw(net,from_layer='conv1',lr_mult=lr_mult,out_layer='conv2',num_output_dw=64,num_output_pw=128,stride = 2)
    ConvDwPw(net,from_layer='conv2',lr_mult=lr_mult,out_layer='conv3',num_output_dw=128,num_output_pw=128,stride = 1)
    ConvDwPw(net,from_layer='conv3',lr_mult=lr_mult,out_layer='conv4',num_output_dw=128,num_output_pw=256,stride = 2)
    ConvDwPw(net,from_layer='conv4',lr_mult=lr_mult,out_layer='conv5',num_output_dw=256,num_output_pw=256,stride = 1)
    ConvDwPw(net,from_layer='conv5',lr_mult=lr_mult,out_layer='conv6',num_output_dw=256,num_output_pw=512,stride = 1)
    ConvDwPw(net,from_layer='conv6',lr_mult=lr_mult,out_layer='conv7',num_output_dw=512,num_output_pw=512,stride = 1)
    ConvDwPw(net,from_layer='conv7',lr_mult=lr_mult,out_layer='conv8',num_output_dw=512,num_output_pw=512,stride = 1)
    ConvDwPw(net,from_layer='conv8',lr_mult=lr_mult,out_layer='conv9',num_output_dw=512,num_output_pw=512,stride = 1)

    ConvDwPw(net,from_layer='conv9',lr_mult=lr_mult,out_layer='conv10',num_output_dw=512,num_output_pw=512,stride = 1)
    ConvDwPw(net,from_layer='conv10',lr_mult=lr_mult,out_layer='conv11',num_output_dw=512,num_output_pw=512,stride = 1)
    ConvDwPw(net,from_layer='conv11',lr_mult=lr_mult,out_layer='conv12',num_output_dw=512,num_output_pw=1024,stride = 2)
    ConvDwPw(net,from_layer='conv12',lr_mult=lr_mult,out_layer='conv13',num_output_dw=1024,num_output_pw=1024,stride = 1)

def bn_scale_relu(bottom, in_place = True):
    bn = L.BatchNorm(bottom, in_place=in_place)
    scale = L.Scale(bn, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def conv_bn_scale_relu(bottom, num_output=32, kernel_size=3, stride=1, pad=1, in_place=True, **kwargs):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, **kwargs)
    conv_relu = bn_scale_relu(conv,in_place=in_place)
    return conv_relu

def conv_relu(bottom, num_output=32, kernel_size=3, stride=1, pad=1, **kwargs):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad, **kwargs)
    relu = L.ReLU(conv, in_place=True)
    return relu

def sdv15(net, from_layer="data"):
    assert from_layer in net.keys()
    kwargs = {'weight_filler': dict(type='xavier'), "bias_term":False}

    net.conv1_1 = conv_bn_scale_relu(net[from_layer], num_output=4, kernel_size=3, stride=1,pad=1,**kwargs)
    net.conv1_2 = conv_bn_scale_relu(net.conv1_1,num_output=4, kernel_size=3, stride=1, pad=1,**kwargs)
    net.pool1 = L.Pooling(net.conv1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = conv_bn_scale_relu(net.pool1, num_output=8, kernel_size=3, stride=1,pad=1,**kwargs)
    net.conv2_2 = conv_bn_scale_relu(net.conv2_1,num_output=8, kernel_size=3, stride=1, pad=1,**kwargs)
    net.pool2 = L.Pooling(net.conv2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = conv_bn_scale_relu(net.pool2, num_output=24, kernel_size=3, stride=1,pad=1,**kwargs)
    net.conv3_2 = conv_bn_scale_relu(net.conv3_1,num_output=24, kernel_size=3, stride=1, pad=1,**kwargs)

    net.conv6_1 = conv_bn_scale_relu(net.conv3_2, num_output=24, kernel_size=3, stride=1,pad=0,**kwargs)
    net.conv6_2 = conv_bn_scale_relu(net.conv6_1,num_output=24, kernel_size=3, stride=2, pad=1,**kwargs)
    
    net.conv7_1 = conv_bn_scale_relu(net.conv6_2, num_output=24, kernel_size=3, stride=1,pad=0,**kwargs)
    net.conv7_2 = conv_bn_scale_relu(net.conv7_1,num_output=24, kernel_size=3, stride=2, pad=1,**kwargs)

    return net

def res10(net, from_layer="data"):
    assert from_layer in net.keys()
    kwargs = {'weight_filler': dict(type='msra'),"bias_term":False}
    net.relu1 = conv_bn_scale_relu(net[from_layer],num_output=32, kernel_size=7, stride=2, pad=3, **kwargs)
    net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    
    net.relu2 = conv_bn_scale_relu(net.pool1, num_output=32, kernel_size=3, stride=1, pad=1, **kwargs)
    net.conv2 = L.Convolution(net.relu2, num_output=32, kernel_size=3, stride=1, pad=1,**kwargs)
    net.merge1 = L.Eltwise(net.pool1, net.conv2)

    net.relu2 = bn_scale_relu(net.merge1, in_place=False)

    net.relu3 = conv_bn_scale_relu(net.relu2, num_output=128, kernel_size=3, stride=2, pad=1, **kwargs)
    net.conv3_2 = L.Convolution(net.relu3, num_output=128, kernel_size=3, stride=1, pad=1,**kwargs)
    net.conv3_3 = L.Convolution(net.relu2, num_output=128, kernel_size=1, stride=2, pad=0,**kwargs)
    net.merge2 = L.Eltwise(net.conv3_2, net.conv3_3)

    net.relu4 = bn_scale_relu(net.merge2, in_place=False)
    net.relu5 = conv_bn_scale_relu(net.relu4, num_output=256, kernel_size=3, stride=2, pad=1, **kwargs)
    net.conv4_2 = L.Convolution(net.relu5, num_output=256, kernel_size=3, stride=1, pad=1,**kwargs)
    net.conv4_3 = L.Convolution(net.relu4, num_output=256, kernel_size=1, stride=2, pad=0,**kwargs)
    net.merge3 = L.Eltwise(net.conv4_2, net.conv4_3)

    net.relu6 = bn_scale_relu(net.merge3, in_place=False)
    net.relu7 = conv_bn_scale_relu(net.relu6, num_output=256, kernel_size=3, stride=1, pad=1, **kwargs)
    net.conv6_2 = L.Convolution(net.relu7, num_output=256, kernel_size=3, stride=1, pad=2,dilation=2,**kwargs)
    net.conv6_3 = L.Convolution(net.relu6, num_output=256, kernel_size=1, stride=1, pad=0,**kwargs)
    net.merge4 = L.Eltwise(net.conv6_2, net.conv6_3)

    net.relu8 = bn_scale_relu(net.merge4, in_place=False)

    net.relu9_1 = conv_relu(net.relu8,num_output=128, kernel_size=1,stride=1,pad=0,**kwargs)
    net.relu9_2 = conv_relu(net.relu9_1,num_output=256, kernel_size=3,stride=2,pad=1,**kwargs)

    net.relu10_1 = conv_relu(net.relu9_2,num_output=64, kernel_size=1,stride=1,pad=0,**kwargs)
    net.relu10_2 = conv_relu(net.relu10_1,num_output=128, kernel_size=3,stride=2,pad=1,**kwargs)

    net.relu11_1 = conv_relu(net.relu10_2,num_output=64, kernel_size=1,stride=1,pad=0,**kwargs)
    net.relu11_2 = conv_relu(net.relu11_1,num_output=128, kernel_size=3,stride=1,pad=1,**kwargs)

    net.relu12_1 = conv_relu(net.relu11_2,num_output=64, kernel_size=1,stride=1,pad=0,**kwargs)
    net.relu12_2 = conv_relu(net.relu12_1,num_output=128, kernel_size=3,stride=1,pad=1,**kwargs)
    return net

def aizoo28(net, from_layer="data"):
    assert from_layer in net.keys()
    kwargs = {'weight_filler': dict(type='xavier'),"bias_term":False}
    net.conv2d_0 = L.Convolution(net[from_layer], num_output=32, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_0 = L.BatchNorm(net.conv2d_0,in_place=True)
    net.relu2d_0 = L.ReLU(net.conv2d_0, in_place=True)
    net.pool2d_0 = L.Pooling(net.relu2d_0, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_1 = L.Convolution(net.pool2d_0, num_output=64, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_1 = L.BatchNorm(net.conv2d_1,in_place=True)
    net.relu2d_1 = L.ReLU(net.conv2d_1, in_place=True)
    net.pool2d_1 = L.Pooling(net.relu2d_1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_2 = L.Convolution(net.pool2d_1, num_output=64, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_2 = L.BatchNorm(net.conv2d_2,in_place=True)
    net.relu2d_2 = L.ReLU(net.conv2d_2, in_place=True)
    net.pool2d_2 = L.Pooling(net.relu2d_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_3 = L.Convolution(net.pool2d_2, num_output=64, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_3 = L.BatchNorm(net.conv2d_3,in_place=True)
    net.relu2d_3 = L.ReLU(net.conv2d_3, in_place=True)
    net.pool2d_3 = L.Pooling(net.relu2d_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_4 = L.Convolution(net.pool2d_3, num_output=128, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_4 = L.BatchNorm(net.conv2d_4,in_place=True)
    net.relu2d_4 = L.ReLU(net.conv2d_4, in_place=True)
    net.pool2d_4 = L.Pooling(net.relu2d_4, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_5 = L.Convolution(net.pool2d_4, num_output=128, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_5 = L.BatchNorm(net.conv2d_5,in_place=True)
    net.relu2d_5 = L.ReLU(net.conv2d_5, in_place=True)
    net.pool2d_5 = L.Pooling(net.relu2d_5, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_6 = L.Convolution(net.pool2d_5, num_output=128, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_6 = L.BatchNorm(net.conv2d_6,in_place=True)
    net.relu2d_6 = L.ReLU(net.conv2d_6, in_place=True)
    net.pool2d_6 = L.Pooling(net.relu2d_6, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_6 = L.Convolution(net.pool2d_5, num_output=128, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_6 = L.BatchNorm(net.conv2d_6,in_place=True)
    net.relu2d_6 = L.ReLU(net.conv2d_6, in_place=True)
    net.pool2d_6 = L.Pooling(net.relu2d_6, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2d_7 = L.Convolution(net.pool2d_6, num_output=128, pad=1, kernel_size=3, stride=1,**kwargs)
    net.conv2d_7 = L.BatchNorm(net.conv2d_7,in_place=True)
    net.relu2d_7 = L.ReLU(net.conv2d_7, in_place=True)

    return net

def CreateMultiBoxHead(net, data_layer="data", num_classes=[], from_layers=[],
        use_objectness=False, normalizations=[], use_batchnorm=True, lr_mult=1,
        use_scale=True, min_sizes=[], max_sizes=[], prior_variance = [0.1],
        aspect_ratios=[], steps=[], img_height=0, img_width=0, share_location=True,
        flip=True, clip=True, offset=0.5, inter_layer_depth=[], kernel_size=1, pad=0,
        conf_postfix='', loc_postfix='',max_out= 0, **bn_param):
    assert num_classes, "must provide num_classes"
    assert num_classes > 0, "num_classes must be positive number"
    if normalizations:
        assert len(from_layers) == len(normalizations), "from_layers and normalizations should have same length"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    if aspect_ratios:
        assert len(from_layers) == len(aspect_ratios), "from_layers and aspect_ratios should have same length"
    if steps:
        assert len(from_layers) == len(steps), "from_layers and steps should have same length"
    net_layers = net.keys()
    assert data_layer in net_layers, "data_layer is not in net's layers"
    if inter_layer_depth:
        assert len(from_layers) == len(inter_layer_depth), "from_layers and inter_layer_depth should have same length"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    objectness_layers = []
    for i in range(0, num):
        from_layer = from_layers[i]

        # Get the normalize value.
        if normalizations:
            if normalizations[i] != -1:
                norm_name = "{}_norm".format(from_layer)
                net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
                from_layer = norm_name

        # Add intermediate layers.
        if inter_layer_depth:
            if inter_layer_depth[i] > 0:
                inter_name = "{}_inter".format(from_layer)
                ConvBNLayer(net, from_layer, inter_name, use_bn=use_batchnorm, use_relu=True, lr_mult=lr_mult,
                      num_output=inter_layer_depth[i], kernel_size=3, pad=1, stride=1, **bn_param)
                from_layer = inter_name

        # Estimate number of priors per location given provided parameters.
        min_size = min_sizes[i]
        if type(min_size) is not list:
            min_size = [min_size]
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]
        max_size = []
        if len(max_sizes) > i:
            max_size = max_sizes[i]
            if type(max_size) is not list:
                max_size = [max_size]
            if max_size:
                assert len(max_size) == len(min_size), "max_size and min_size should have same length."
        if max_size:
            num_priors_per_location = (2 + len(aspect_ratio)) * len(min_size)
        else:
            num_priors_per_location = (1 + len(aspect_ratio)) * len(min_size)
        if flip:
            num_priors_per_location += len(aspect_ratio) * len(min_size)
        step = []
        if len(steps) > i:
            step = steps[i]

        # Create location prediction layer.
        name = "{}_mbox_loc{}".format(from_layer, loc_postfix)
        num_loc_output = num_priors_per_location * 4
        if not share_location:
            num_loc_output *= num_classes
        if max_out>0:
          slice_point = []
          for i in range(max_out-1):
            slice_point.append(num_loc_output*(i+1))
          ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output*max_out, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
          net[name+'_slice_0'],net[name+'_slice_1'],net[name+'_slice_2'],net[name+'_slice_3'],\
          net[name+'_slice_4'],net[name+'_slice_5'] ,\
          net[name+'_slice_6'],net[name+'_slice_7'] ,\
          net[name+'_slice_8'] = L.Slice(net[name],slice_param={'axis':1,'slice_point':slice_point},ntop = 9)
          net[name+'_max'] = L.Eltwise(net[name+'_slice_0'],net[name+'_slice_1'],net[name+'_slice_2'],net[name+'_slice_3'],\
          net[name+'_slice_4'],net[name+'_slice_5'] ,\
          net[name+'_slice_6'],net[name+'_slice_7'] ,\
          net[name+'_slice_8'],eltwise_param = {'operation':2})
          name = name+'_max'
        else:
          ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_loc_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        loc_layers.append(net[flatten_name])

        # Create confidence prediction layer.
        name = "{}_mbox_conf{}".format(from_layer, conf_postfix)
        num_conf_output = num_priors_per_location * num_classes
        if max_out>0:
          ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output*max_out, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
          slice_point = []
          for i in xrange(max_out-1):
            slice_point.append(num_conf_output*(i+1))
          net[name+'_slice_0'],net[name+'_slice_1'],net[name+'_slice_2'],net[name+'_slice_3'],\
          net[name+'_slice_4'],net[name+'_slice_5'] ,\
          net[name+'_slice_6'],net[name+'_slice_7'] ,\
          net[name+'_slice_8'] = L.Slice(net[name],slice_param={'axis':1,'slice_point':slice_point},ntop = 9)
          net[name+'_max'] = L.Eltwise(net[name+'_slice_0'],net[name+'_slice_1'],net[name+'_slice_2'],net[name+'_slice_3'],\
          net[name+'_slice_4'],net[name+'_slice_5'] ,\
          net[name+'_slice_6'],net[name+'_slice_7'] ,\
          net[name+'_slice_8'],eltwise_param = {'operation':2})
          name = name+'_max'
        else:
          ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
            num_output=num_conf_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
        permute_name = "{}_perm".format(name)
        net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
        flatten_name = "{}_flat".format(name)
        net[flatten_name] = L.Flatten(net[permute_name], axis=1)
        conf_layers.append(net[flatten_name])

        # Create prior generation layer.
        name = "{}_mbox_priorbox".format(from_layer)
        net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_size,
                clip=clip, variance=prior_variance, offset=offset)
        if max_size:
            net.update(name, {'max_size': max_size})
        if aspect_ratio:
            net.update(name, {'aspect_ratio': aspect_ratio, 'flip': flip})
        if step:
            net.update(name, {'step': step})
        if img_height != 0 and img_width != 0:
            if img_height == img_width:
                net.update(name, {'img_size': img_height})
            else:
                net.update(name, {'img_h': img_height, 'img_w': img_width})
        priorbox_layers.append(net[name])

        # Create objectness prediction layer.
        if use_objectness:
            name = "{}_mbox_objectness".format(from_layer)
            num_obj_output = num_priors_per_location * 2
            ConvBNLayer(net, from_layer, name, use_bn=use_batchnorm, use_relu=False, lr_mult=lr_mult,
                num_output=num_obj_output, kernel_size=kernel_size, pad=pad, stride=1, **bn_param)
            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            objectness_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc"
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf"
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_priorbox"
    net[name] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net[name])
    if use_objectness:
        name = "mbox_objectness"
        net[name] = L.Concat(*objectness_layers, axis=1)
        mbox_layers.append(net[name])

    return mbox_layers

def build_backbone(name):
    nets = { 
        "mobilenetv1": MobileNetBody,
        "sdv15": sdv15,
        "res10": res10,
        "aizoo28": aizoo28
    }
    return nets[name]

if __name__=="__main__":
    res10 = build_backbone("res10")
    batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        }
    ]
    annotated_data_param = {
        'label_map_file': "data/Face/labelmap.prototxt",
        'batch_sampler': batch_sampler,
        }
    test_transform_param = {
        'mean_value': [127.5,127.5,127.5],
        'scale': 0.007843,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': 160,
                'width': 90,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
    from caffe.proto import caffe_pb2
    kwargs = {
                'include': dict(phase=caffe_pb2.Phase.Value('TEST')),
                'transform_param': test_transform_param,
                }
    net = caffe.NetSpec()
    net.data = L.AnnotatedData(name="data", annotated_data_param=annotated_data_param,
        data_param=dict(batch_size=16, backend=P.Data.LMDB, source="data/Face/ldmb/train_lmdb"),
        ntop=1, **kwargs)
    net = res10(net)
    with open("output/model.prototxt", 'w') as f:
        f.write(str(net.to_proto()))