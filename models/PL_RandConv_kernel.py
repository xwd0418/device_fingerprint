from models.PL_resnet import *
from models.model_factory import *
import torchvision
from models.dc1d.dc1d.nn import DeformConv1d
import scipy, time
import matplotlib.pyplot as plt


class RandConv(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.mix_original_coeff = config['experiment']['mix_original_coeff']
        self.consistency_loss_coeff = config['experiment']['consistency_loss_coeff']
        if self.consistency_loss_coeff:
            self.consistency_loss_func = nn.KLDivLoss()
        
        
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        # print(x.shape, "\n\n\n\n\n")
        num_random_convs = 3 # some magic number from the paper
        logits_recorder = []
        total_loss = 0
        consistency_loss = 0
        for _ in range(num_random_convs):
            with torch.no_grad: 
                random_layer = nn.Conv1d(2,2,kernel_size=self.config['experiment']['kernel_size'])
                rand_conved_x = random_layer(x)
                if self.mix_original_coeff:
                    rand_conved_x = (1-self.mix_original_coeff)*rand_conved_x + self.mix_original_coeff*x
            logits = self(rand_conved_x)
            logits_recorder.append(logits)
            loss = self.criterion(logits, y)
            total_loss += loss
            preds = torch.argmax(logits, dim=1)     
            self.train_accuracy(preds, y)
            
        mean_logits = (logits_recorder[0] + logits_recorder[1] + logits_recorder[2])/3
        for i in range(num_random_convs):
            consistency_loss += self.consistency_loss_func(logits_recorder[i],mean_logits)    
        consistency_loss /= 3     
        total_loss  += self.consistency_loss_coeff * consistency_loss
            
        self.log("train/loss", total_loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        return total_loss
    