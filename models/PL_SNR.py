from models.PL_resnet import *
from models.resnet_SNR  import resnet18_snr_causality
from models.model_factory import *



class SNR(Baseline_Resnet):
    def __init__(self,config):
        super().__init__(config)
        self.num_classes = 150
        self.network = resnet18_snr_causality(pretrained=False, num_classes=self.num_classes)
        
        self.network.train()
        in_channels = 2
        self.network.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)            

        self.network.bn_eval()
        self.best_accuracy_val = -1
        
    # def on_train_epoch_start(self) :
        
    def training_step(self, batch, batch_idx):
        # return super().training_step(batch, batch_idx)
        for weight in self.network.parameters():
                    weight.fast = None
                    
        x, y = self.unpack_batch(batch)
        
        # forward with the original parameters
        outputs, _, \
        x_IN_1_prob, x_1_useful_prob, x_1_useless_prob, \
        x_IN_2_prob, x_2_useful_prob, x_2_useless_prob, \
        x_IN_3_prob, x_3_useful_prob, x_3_useless_prob, \
        x_IN_3_logits, x_3_useful_logits, x_3_useless_logits = self.network(x=x)
        
        # Causality loss:
        loss_causality =  self.get_causality_loss(self.get_entropy(x_IN_1_prob), self.get_entropy(x_1_useful_prob), self.get_entropy(x_1_useless_prob)) + \
                         self.get_causality_loss(self.get_entropy(x_IN_2_prob), self.get_entropy(x_2_useful_prob), self.get_entropy(x_2_useless_prob)) + \
                         self.get_causality_loss(self.get_entropy(x_IN_3_prob), self.get_entropy(x_3_useful_prob), self.get_entropy(x_3_useless_prob)) + \
                         self.criterion(x_3_useful_logits, y)
        # CE loss
        ce_loss = self.criterion(outputs, y)
        
        loss = ce_loss + loss_causality * self.config['experiment']['causality_coeff']
        preds = torch.argmax(outputs, dim=1)
        
        self.train_accuracy(preds, y)
        self.log("train/total_loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/loss_causality", loss_causality,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        # self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                #  on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   


        return loss
          
    def validation_step(self, batch, batch_idx):   
        
        x, y = self.unpack_batch(batch)
        tuples = self.network(x)

        preds = tuples[1]['Predictions']
        self.val_accuracy(preds, y)
        # self.log("val/loss", loss, prog_bar=False, sync_dist=torch.cuda.device_count()>1)
        self.log("val/acc", self.val_accuracy, prog_bar=True, sync_dist=torch.cuda.device_count()>1)
        
        self.bn_process()
    
    
    def test_step(self, batch, batch_idx):   
        
        x, y = self.unpack_batch(batch)
        tuples = self.network(x)

        preds = tuples[1]['Predictions']
        self.test_accuracy(preds, y)
        # self.log("val/loss", loss, prog_bar=False, sync_dist=torch.cuda.device_count()>1)
        self.log("test/acc", self.test_accuracy, prog_bar=True, sync_dist=torch.cuda.device_count()>1)
        
               
    def bn_process(self):    
        self.network.bn_eval()

 
    def get_entropy(self, p_softmax):
        # exploit ENTropy minimization (ENT) to help DA,
        mask = p_softmax.ge(0.000001)
        mask_out = torch.masked_select(p_softmax, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return (entropy / float(p_softmax.size(0)))

    def get_causality_loss(self, x_IN_entropy, x_useful_entropy, x_useless_entropy):
        self.ranking_loss = torch.nn.SoftMarginLoss()
        y = torch.ones_like(x_IN_entropy)
        return self.ranking_loss(x_IN_entropy - x_useful_entropy, y) + self.ranking_loss(x_useless_entropy - x_IN_entropy, y)
