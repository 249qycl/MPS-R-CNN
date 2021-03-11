import math
import os
import sys

import torch
import yaml
from datasets.loader import DataModule
from losses.sparse_rcnn_loss import SparseRCNNLoss
from metrics.map import coco_map
from nets.sparse_rcnn import SparseRCNN, post_process
from torch import nn
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
from utils.model_utils import AverageLogger, rand_seed, reduce_sum
from utils.optim_utils import split_optimizer_v2

rand_seed(1024)

class SparseRCNNSolver(object):
    def __init__(self, cfg_path):
        super(SparseRCNNSolver, self).__init__()
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.data_cfg = cfg['data']
        self.model_cfg = cfg['model']
        self.optim_cfg = cfg['optim']
        self.val_cfg = cfg['val']
        #打印配置字典
        self.device = torch.device("cuda:0")
        self.train_loader, self.val_loader = DataModule(
            **self.data_cfg).loader()
        self.model = SparseRCNN(**self.model_cfg).to(self.device)
        self.criterion = self.configure_criterion()
        self.scaler = amp.GradScaler(enabled=True) if self.optim_cfg['amp'] else None
        self.optimizer = split_optimizer_v2(self.model, self.optim_cfg)
        self.scheduler = self.configure_scheduler()
        self.ema_model = self.configure_ema(self.model, decay=0.999)
        self.model_load()
        #logger
        self.cls_loss_logger = AverageLogger()
        self.l1_loss_logger = AverageLogger()
        self.iou_loss_logger = AverageLogger()
        self.match_num_logger = AverageLogger()
        self.loss_logger = AverageLogger()

    def configure_scheduler(self):
        scheduler = OneCycleLR(self.optimizer,
                               max_lr=self.optim_cfg['lr'],
                               pct_start=self.optim_cfg['warm_up_percent'],
                               steps_per_epoch=len(self.train_loader),
                               epochs=self.optim_cfg['epochs'])
        return scheduler
    
    def configure_criterion(self):
        criterion=SparseRCNNLoss(iou_type=self.model_cfg['iou_type'],
                                   iou_weights=self.model_cfg['iou_weights'],
                                   iou_cost=self.model_cfg['iou_cost'],
                                   cls_weights=self.model_cfg['cls_weights'],
                                   cls_cost=self.model_cfg['cls_cost'],
                                   l1_weights=self.model_cfg['l1_weights'],
                                   l1_cost=self.model_cfg['l1_cost'])
        return criterion

    def configure_ema(self, model, decay=0.999):
        decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))
        avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:\
                    averaged_model_parameter*decay_fn(num_averaged)+\
                    (1.-decay_fn(num_averaged))*model_parameter
        ema = AveragedModel(model, avg_fn=avg_fn)
        return ema

    def model_load(self):
        directory = self.val_cfg['weight_path']
        weight_path = directory + '/' + 'sparse_rcnn_resnet50_last.pth'
        if self.data_cfg['load_model'] and os.listdir(directory):
            state_dict = torch.load(weight_path)
            self.model.load_state_dict(state_dict['model'])
            # self.optimizer.load_state_dict(state_dict['optim'])#消耗显存
            self.ema_model.load_state_dict(state_dict['ema'])
            self.best_map = state_dict['map'] / 100
            self.last_epoch = state_dict['epoch']
        else:
            self.last_epoch =0
            self.best_map =0.0

    def logger(self, epoch,h, loss, iou_loss, l1_loss, cls_loss, match_num):
        lr = self.optimizer.param_groups[0]['lr']
        self.loss_logger.update(loss.item())
        self.iou_loss_logger.update(iou_loss.item())
        self.l1_loss_logger.update(l1_loss.item())
        self.cls_loss_logger.update(cls_loss.item())
        self.match_num_logger.update(match_num)
        str_template = \
            "epoch:{:2d}|match_num:{:0>4d}|size:{:3d}|loss:{:6.4f}|cls:{:6.4f}|l1:{:6.4f}|iou:{:6.4f}|lr:{:8.6f} "
        return str_template.format(epoch + 1, int(match_num), h,
                                   self.loss_logger.avg(),
                                   self.cls_loss_logger.avg(),
                                   self.l1_loss_logger.avg(),
                                   self.iou_loss_logger.avg(), lr)

    def train(self, epoch):
        self.loss_logger.reset()
        self.cls_loss_logger.reset()
        self.l1_loss_logger.reset()
        self.iou_loss_logger.reset()
        self.match_num_logger.reset()
        self.model.train()

        pbar = tqdm(self.train_loader)
        for i, (img_tensor, targets_tensor, batch_len) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with amp.autocast(enabled=True):
                    cls_predicts, box_predicts, shapes = self.model(img_tensor)
                    targets = {"target": targets_tensor, "batch_len": batch_len}
                    cls_loss, iou_loss, l1_loss, match_num = self.criterion(cls_predicts, box_predicts, targets, shapes[0])
                    loss = cls_loss + l1_loss + iou_loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
            self.ema_model.update_parameters(self.model)
            pbar.set_description(self.logger(epoch,h, loss, iou_loss, l1_loss,cls_loss, match_num))

    @torch.no_grad()
    def val(self, epoch):
        predict_list = list()
        target_list = list()
        self.model.eval()
        self.ema_model.eval()

        pbar = tqdm(self.val_loader)
        for img_tensor, targets_tensor, batch_len in pbar:
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            cls_predicts, box_predicts, shapes = self.ema_model(img_tensor)
            predicts = post_process(cls_predicts[-1],box_predicts[-1],shapes,**self.model_cfg)
            for pred, target in zip(predicts, targets_tensor.split(batch_len)):
                predict_list.append(pred)
                target_list.append(target)

        mp, mr, map50, mean_ap = coco_map(predict_list, target_list)
        mp = reduce_sum(torch.tensor(mp, device=self.device))
        mr = reduce_sum(torch.tensor(mr, device=self.device))
        map50 = reduce_sum(torch.tensor(map50, device=self.device))
        mean_ap = reduce_sum(torch.tensor(mean_ap, device=self.device))
        print("*" * 20, "eval start", "*" * 20)
        print("epoch: {:2d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}".
              format(epoch + 1, mp * 100, mr * 100, map50 * 100, mean_ap * 100))
        print("*" * 20, "eval end", "*" * 20)
        self.model_save(mean_ap,epoch)

    def model_save(self,mean_ap,epoch):
        last_weight_path = os.path.join(
            self.val_cfg['weight_path'],
            "{:s}_{:s}_last.pth".format(self.cfg['model_name'],
                                        self.model_cfg['backbone']))
        best_map_weight_path = os.path.join(
            self.val_cfg['weight_path'],
            "{:s}_{:s}_best_map.pth".format(self.cfg['model_name'],
                                            self.model_cfg['backbone']))
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optimizer.state_dict(),
            "map": mean_ap * 100,
            "epoch": epoch,
            "ema": self.ema_model.state_dict()  # 可以加载参数
        }
        torch.save(ckpt, last_weight_path)
        if mean_ap > self.best_map:
            torch.save(ckpt, best_map_weight_path)
            self.best_map = mean_ap

    def run(self):
        for epoch in range(self.optim_cfg['epochs']):
            epoch += self.last_epoch
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.val(epoch)
