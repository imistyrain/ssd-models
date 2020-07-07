from yacs.config import CfgNode as CN

cfg = CN()
cfg.model = CN()
cfg.model.name = 'svd15'
cfg.model.mbox_source_layers = ['conv3_2', 'conv6_2', 'conv7_2']
cfg.model.min_ratio = 20
cfg.model.max_ratio = 90
cfg.model.min_size = 120
cfg.model.steps = [4,8,16]
cfg.model.aspect_ratios = [[],[],[2]]
cfg.model.normalizations = [20, -1, -1]

cfg.dataset = CN()
cfg.dataset.name = "voc"
cfg.dataset.num_classes = 21
cfg.dataset.num_test_image = 285
cfg.dataset.train = "trainval"
cfg.dataset.val = "test"

def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()

if __name__=="__main__":
    config_path = "cfgs/voc.yml"
    load_config(cfg,config_path)
    print(cfg)