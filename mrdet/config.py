from yacs.config import CfgNode as CN

_C = CN()
_C.model = CN()
_C.model.name = "sdv15"
_C.model.width = 160
_C.model.height = 90
_C.model.mbox_source_layers = ['conv3_2', 'conv6_2', 'conv7_2']
_C.model.min_ratio = 20
_C.model.max_ratio = 90
_C.model.min_size = 120
_C.model.steps = [4, 8, 16]
_C.model.aspect_ratios = [[],[],[2]]
_C.model.normalizations =  [20, -1, -1]

_C.dataset = CN()
_C.dataset.name = "Face"
_C.dataset.num_classes = 2
_C.dataset.num_test_image = 2000
_C.dataset.train = "trainval"
_C.dataset.val = "test"

_C.opt = CN()
_C.opt.lr = 0.001
_C.opt.batch_size = 16
_C.opt.accum_batch_size = 64
_C.opt.test_batch_size = 8

def get_cfg_default():
    return _C.clone()

if __name__=="__main__":
    cfg = get_cfg_default()
    cfg.merge_from_file("cfgs/Face/sdv15/160x90.yml")
    cfg.freeze()
    print(cfg)