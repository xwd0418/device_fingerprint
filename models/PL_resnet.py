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
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from models.resnet import resnet18

class Baseline_Resnet(PL.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # self.batch_size = config['dataset']['batch_size']
        self.num_classes = 150
        self.encoder = self.get_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        
    def forward(self, x, feat=False):
        x = self.encoder(x, feat=feat)
        return x

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False)
        self.log("train/acc", self.train_accuracy, prog_bar=True)   

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val/loss", loss, prog_bar=False)
        self.log("val/acc", self.val_accuracy, prog_bar=True)

        
    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test/loss", loss, prog_bar=False)
        self.log("test/acc", self.test_accuracy, prog_bar=True)
        
    def unpack_batch(self, batch, need_date=False):
        # x,y,date = zip(*batch)   
        # x,y,date = torch.cat(x), torch.cat(y),torch.cat(date) 
        x,y,date = batch
        if need_date:
            return x,y,date
        else:
            return x,y
    
    def configure_optimizers(self):
     
        # print(self.parameters())
        if self.config['experiment'].get('optimizer') == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),lr =1, momentum=0.9, weight_decay=0.005, nesterov=True)
        else:
            optimizer = torch.optim.Adam(self.parameters(),
                                    lr= self.config['experiment']['learning_rate'],
                                    weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
        return {'optimizer': optimizer,"lr_scheduler":scheduler, "monitor":"val/loss"}

    def train_dataloader(self):
        return DataLoader(
            self.domained_data[2], # ConcatDataset(self.df_data_train),
            batch_size=self.config['dataset']['batch_size']//3,
            num_workers=32,
            # pin_memory=True
        )


    def val_dataloader(self):
        return DataLoader(
            self.domained_data[0],# ConcatDataset(self.df_data_val),
            batch_size=self.config['dataset']['batch_size']//3,
            num_workers=32,
            # pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(self.df_data_test, 
                          batch_size=self.config['dataset']['batch_size'], 
                          num_workers = 32)
    
    def setup(self, stage = None):  
        self.get_all()
        self.df_data_train, self.df_data_val = [],[]
        for i in range(3):
            val_num = len(self.domained_data[i])//10
            t,v = random_split(self.domained_data[i], [len(self.domained_data[i])-val_num,val_num])
            self.df_data_train.append(t)
            self.df_data_val.append(v)
        self.df_data_test = self.domained_data[3]
            
    def get_model(self):
        model = resnet18(pretrained=False, num_classes=self.num_classes)
        in_channels = 2
        model.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)            
        return model
            
    def get_all(self) :
        pickleFile = open("/root/dataset/ManyTx.pkl","rb")
        all_info = pickle.load(pickleFile)
        data = all_info['data']

        self.domained_data = [],[],[],[]
        for label in range(len(data)):
            # one_hot_encoded = F.one_hot(torch.tensor([label]), num_classes=len(data))
            for i in data[label]:
                for date, j in enumerate(i):
                    for k in j[1]:
                        self.domained_data[date].append((k.T.astype("float32"),label, date)) 
                        
    
    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1000.0):
        kick_in_iter = self.config['experiment'].get('adv_coeff_kick_in_iter')
        if kick_in_iter:
            coeff_param = kick_in_iter/10
        else : 
            coeff_param = 20.0
        return np.float(coeff_param* (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        
        # print(len(combined))
        # return zip(*combined)
        # unzipped_data, unzipped_label, unzipped_date = zip(*combined)
        # return np.concatenate(unzipped_data), np.array(unzipped_label), np.array(unzipped_date)

    def __len__(self):
        return min(len(d) for d in self.datasets)                

