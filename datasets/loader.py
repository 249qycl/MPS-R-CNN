from torch.utils.data import random_split
from .coco import COCODataSets,DataLoaderX

class DataModule(object):
    def __init__(self,**cfg):
        train_data = COCODataSets(img_root=cfg['train_img_root'],
                                  annotation_path=cfg['train_annotation_path'],
                                  max_thresh=cfg['max_thresh'],
                                  use_crowd=cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=cfg['remove_blank']
                                  )
        val_data = COCODataSets(img_root=cfg['val_img_root'],
                                  annotation_path=cfg['val_annotation_path'],
                                  max_thresh=cfg['max_thresh'],
                                  use_crowd=cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        if cfg['debug']==True:
            train_data,_=random_split(train_data,[len(train_data)*cfg['debug_ratio'],len(train_data)-len(train_data)*cfg['debug_ratio']])
            val_data,_=random_split(val_data,[len(val_data)*cfg['debug_ratio'],len(val_data)-len(val_data)*cfg['debug_ratio']])
        self.train_data = train_data
        self.val_data = val_data
        
        self.train_loader = DataLoaderX(dataset=train_data,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  collate_fn=train_data.collect_fn,
                                  shuffle=True, pin_memory=True)
        self.val_loader = DataLoaderX(dataset=val_data,
                                  batch_size=cfg['batch_size'],
                                  num_workers=cfg['num_workers'],
                                  collate_fn=val_data.collect_fn, pin_memory=True)
    def loader(self):
        print(f"train_data: {len(self.train_data)} | val_data: {len(self.val_data)} | empty_data: {self.train_data.empty_images_len}")
        print(f"train_iter: {len(self.train_loader)} | val_iter: {len(self.val_loader)}")
        return self.train_loader,self.val_loader
