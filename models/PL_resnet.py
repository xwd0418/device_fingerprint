import os,sys
from glob import glob
import pickle
import torch, copy
import numpy as np
import random
from tqdm import tqdm
import pytorch_lightning as PL
import pandas as pd
from IPython.display import display
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
import torchvision.models 
from models.resnet import resnet18

class Baseline_Resnet(PL.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
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
        
    def unpack_batch(self, batch):
        x, y = batch
        return x,y
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['experiment']['learning_rate'])
    
    def train_dataloader(self):
        return DataLoader(self.df_data_val, batch_size=self.config['dataset']['batch_size'], num_workers = 32)

    def val_dataloader(self):
        return DataLoader(self.df_data_val, batch_size=self.config['dataset']['batch_size'], num_workers = 32)

    def test_dataloader(self):
        return DataLoader(self.df_data_test, batch_size=self.config['dataset']['batch_size'], num_workers = 32)
    
    def setup(self, stage = None):
        source_data,target_data = self.get_all()
        val_num = len(source_data)//10
        self.df_data_train, self.df_data_val = random_split(source_data, [len(source_data)-val_num,val_num])
        self.df_data_test = target_data
            
    def get_model(self):
        model = resnet18(pretrained=False, num_classes=self.num_classes)
        in_channels = 2
        model.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)            
        return model
            
    def get_all(self) :
        pickleFile = open("/root/device_fingerprint/dataset/ManyTx.pkl","rb")
        all_info = pickle.load(pickleFile)
        data = all_info['data']

        source_data,target_data = [],[]
        for label in range(len(data)):
            # one_hot_encoded = F.one_hot(torch.tensor([label]), num_classes=len(data))
            for i in data[label]:
                for j in i[0:3]:
                    for k in j[1]:
                        source_data.append((k.T.astype("float32"),label)) 
                for j in [ i[3] ]: # this seems dumb, just for sake of pretty alignment
                    for k in j[1]: # delete this line if we need to put the 50 together
                        target_data.append((k.T.astype("float32"),label)) 
                        # shape of k is 256 * 2
        return source_data,target_data 
    
    def calc_coeff(self, iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=1000.0):
        kick_in_iter = self.config['experiment'].get('adv_coeff_kick_in_iter')
        if kick_in_iter:
            coeff_param = kick_in_iter/10
        else : 
            coeff_param = 20.0
        return np.float(coeff_param* (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

                

