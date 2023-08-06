from models.PL_resnet import *
from models.model_factory import *

from models.dc1d.dc1d.nn import DeformConv1d

class RandConv(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
    
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        
        rand_conved_x = self.rand_conv(x)
        logits = self(rand_conved_x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        return loss
    
    def rand_conv(self, x):
        if self.config['model']['conv_type'] == "plain":
            rand_conv_layer = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, padding="same")
        elif self.config['model']['conv_type'] == "deformabale":
            rand_conv_layer = DeformConv1d(in_channels=2, out_channels=2, kernel_size=3, padding="same")
        else :
            raise Exception("what kind of random conv to use?")
        
        # initialize random weights 
        rand_weights = np.random.normal(loc=0.0, scale=1.0, size=rand_conv_layer.weight.shape)
        gaussian_filter = np.array([0.25, 0.5, 0.25])
        rand_weights = rand_weights * gaussian_filter
        rand_weights = torch.tensor(rand_weights)
        rand_conv_layer.weight = torch.nn.Parameter(rand_weights)
        rand_conv_layer.to(x)
        
        for _ in range(self.config['model']["conv_reptitions"]):
            x = rand_conv_layer(x)
              
        
        
        del rand_conv_layer
        return x