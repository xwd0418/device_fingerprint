import os,sys
from glob import glob
import pickle
import torch, copy
import numpy as np
import random
from tqdm import tqdm
import pytorch_lightning as PL
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from models.resnet import resnet18

class Baseline_Resnet(PL.LightningModule):
    def __init__(self,config):
        super().__init__()
        print("model initializing")
        self.config = config
        print(config)
        self.num_classes = 150
        self.get_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_log_on_epoch = True
        self.test_result = -1
        
    def forward(self, x, feat=False):
        x = self.encoder(x, feat=feat)
        return x

    # def on_train_epoch_start(self) -> None:
    #     super().on_train_epoch_start()
    #     # random.shuffle(self.trainer.datamodule.df_data_train[0])
    #     for d in self.trainer.datamodule.df_data_train:
    #         random.shuffle(d)
    #     print("traing loader is shuffled")
    
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        # if batch_idx == 1:
        #     print("date1: ", x[0,0,0])    
        #     print("date3: ", x[-1,0,0])    
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        return loss
    
    # def training_step_end(self, training_step_outputs):
    #     return {'loss': training_step_outputs['loss'].sum()}
    
    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, single_domain_loader=True)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val/loss", loss, prog_bar=False, sync_dist=True)
        self.log("val/acc", self.val_accuracy, prog_bar=True, sync_dist=True)

        
    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, single_domain_loader=True)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test/loss", loss, prog_bar=False, sync_dist=True)
        self.log("test/acc", self.test_accuracy, prog_bar=True, sync_dist=True)
        # return self.test_accuracy.compute()
        # return 0
      
            
    def unpack_batch(self, batch, need_date=False, single_domain_loader = False):
        if single_domain_loader:
            x,y,date = batch
        else:
            x,y,date = zip(*batch)   
            x,y,date = torch.cat(x), torch.cat(y),torch.cat(date) 
            
        if need_date:
            return x,y,date
        else:
            return x,y
    
    def configure_optimizers(self):
     
        # print(self.parameters())
        if self.config['experiment'].get('optimizer') == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr =self.config['experiment']['learning_rate'],
                                        momentum=self.config['experiment']['momentum'],
                                        weight_decay=self.config['experiment']['weight_decay'], 
                                        nesterov=self.config['experiment']['nesterov']=="True")
        elif self.config['experiment'].get('optimizer') == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                    lr= self.config['experiment']['learning_rate'],
                                    weight_decay=self.config['experiment']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer,"lr_scheduler":scheduler, "monitor":"val/loss"}
        
    def get_model(self):
        model = resnet18(pretrained=False, num_classes=self.num_classes)
        in_channels = 2
        model.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)            
        self.encoder =  model                      
        
        load_from = self.config['experiment'].get("load_from")
        if load_from:
            checkpoint = torch.load(load_from)
            print(f'loaded from {load_from}')
            self.load_state_dict(checkpoint['state_dict']) 
        elif self.config['use_pretrained']:
            model_path = '/root/ckpts/resnet_all_58_accu.ckpt'
            checkpoint = torch.load(model_path)
            print('successfully loaded')
            self.load_state_dict(checkpoint['state_dict'])
            
    
    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1000.0, kick_in_iter=0):
        
        iter_num = max(iter_num-kick_in_iter, 0)
        return float(2* (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


    def give_opt_and_sch(self):
        if type(self.lr_schedulers()) is not list:
            yield self.optimizers(), self.lr_schedulers()
        else:
            for i in range(len(self.lr_schedulers())):
                yield  self.optimizers()[i], self.lr_schedulers()[i]
    def give_opt(self):
        if type(self.lr_schedulers()) is not list:
            yield self.optimizers()
        else:
            for i in range(len(self.lr_schedulers())):
                yield  self.optimizers()[i]
    def give_sch(self):
        if type(self.lr_schedulers()) is not list:
            yield self.lr_schedulers()
        else:
            for i in range(len(self.lr_schedulers())):
                yield  self.lr_schedulers()[i]