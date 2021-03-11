import netron
import torch
from torch import nn
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

from nets.common import FrozenBatchNorm2d
from nets.sparse_rcnn import SparseRCNN


default_cfg = {
    "in_channel": 256,
    "inner_channel": 64,
    "num_cls": 7,#80
    "dim_feedforward": 2048,
    "nhead": 8,
    "dropout": 0,
    "pooling_resolution": 7,
    "activation": nn.ReLU,
    "cls_tower_num": 1,
    "reg_tower_num": 3,
    "num_heads": 6,
    "return_intermediate": True,
    "num_proposals": 100,
    "backbone": "resnet50",
    "pretrained": True,
    "norm_layer": FrozenBatchNorm2d,
    # loss cfg
    "iou_type": "giou",
    "iou_weights": 2.0,
    "iou_cost": 1.0,
    "cls_weights": 2.0,
    "cls_cost": 1.0,
    "l1_weights": 5.0,
    "l1_cost": 1.0
}

if __name__ == '__main__':
    
    images = torch.rand((1, 3, 100, 100))
    net = SparseRCNN(**default_cfg)
