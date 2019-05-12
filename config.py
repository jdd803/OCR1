#import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# network related params
config.network = edict()
config.network.pretrained = './model/pretrained_model/resnet_v2_101'
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.IMAGE_STRIDE = 0
config.network.RPN_FEAT_STRIDE = 8
config.network.RCNN_FEAT_STRIDE = 8
config.network.PSROI_BINS = 3
config.network.FIXED_PARAMS = ['conv0', 'stage1', 'gamma', 'beta']
config.network.FIXED_PARAMS_SHARED = ['conv0', 'stage1', 'stage2', 'stage3', 'gamma', 'beta']
config.network.ANCHOR_SCALES = (2, 4, 8, 16)
config.network.ANCHOR_RATIOS = (0.2, 0.5, 2, 5)
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)


# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = 'SDS_train'
config.dataset.test_image_set = 'SDS_val'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
config.dataset.NUM_CLASSES = 1

# Training configurations
config.TRAIN = edict()
config.TRAIN.lr = 0.0005
config.TRAIN.lr_step = ''
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''

# whether resume training
config.TRAIN.RESUME = False

# whether flip image
config.TRAIN.FLIP = True

# whether shuffle image
config.TRAIN.SHUFFLE = True

# whether use OHEM
config.TRAIN.ENABLE_OHEM = False

# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2

# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False

# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.BATCH_ROIS_OHEM = 128

# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0

# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256

# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False

# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 6000
config.TRAIN.RPN_POST_NMS_TOP_N = 200
config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
config.TRAIN.RPN_ALLOWED_BORDER = 0
config.TRAIN.USE_GPU_NMS = False

# whether select from all rois or bg rois
config.TRAIN.GAP_SELECT_FROM_ALL = True
config.TRAIN.IGNORE_GAP = False

# binary_threshold for proposal annotator
config.TRAIN.BINARY_THRESH = 0.4

# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

# loss weight for cls, bbox and mask
config.TRAIN.LOSS_WEIGHT = (1.0, 10.0, 1.0)
config.TRAIN.CONVNEW3 = False

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = True

# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

# Test Model Epoch
config.TEST.test_epoch = 0

# TEST iteration
config.TEST.ITER = 1
config.TEST.MIN_DROP_SIZE = 16

# mask merge
config.TEST.USE_MASK_MERGE = False
config.TEST.USE_GPU_MASK_MERGE = False
config.TEST.MASK_MERGE_THRESH = 0.5

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        #exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")