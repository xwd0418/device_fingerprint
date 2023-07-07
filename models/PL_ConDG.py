from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import mmd
from models.loss_module.domain_adv_loss import DALoss


class ConDG(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.num_classes = 150
        self.condition_domain_classifiers =  
                [   ConDomainClassifier(in_feat=512*8, out_class=4, hidden=[256]) 
                    for i in range(self.num_classes) ]

        
    def training_step(self, batch, batch_idx):
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)

        # conditional domain classifier
        if self.config['experiment']['con_domain_coeff']:
            con_domain_loss = 0
            for i in range(self.num_class):
                idx = y==i
                con_domain_loss += condition_domain_classifiers[i](feature[idx], y[idx], date[idx]) 
        loss += self.config['experiment']['con_domain_coeff'] * con_domain_loss
    
        # weighted domain classifier
        if self.config['experiment']['weighted_domain_coeff']:
            weighted_domain_loss = 0
            ??
        loss += self.config['experiment']['weighted_domain_coeff'] * weighted_domain_loss
        
        
        return loss
     
    def validation_step(self, batch, batch_idx):
        ?
        
        if self.config['experiment']['adv_coeff']:
            coeff = self.calc_coeff(self.global_step)
            feat = feature.view(len(feature), -1)
            prior_feat = torch.tensor(np.random.laplace(loc=0,scale=0.1,size=feat.shape)).float()
            prior_feat = prior_feat.to(feat)
            feature_together = torch.cat([feat, prior_feat], dim=0)
            domain_prediction = self.discriminator(feature_together, coeff)
            adv_loss = self.adv_criterion(domain_prediction,coeff)
            loss += self.config['experiment']['adv_coeff']*adv_loss
            self.log("val/domain_prediction_feat", torch.mean(domain_prediction[0:feat.shape[0]]) , sync_dist=self.train_log_on_epoch)
            self.log("val/domain_prediction_prior", torch.mean(domain_prediction[feat.shape[0]:]) , sync_dist=self.train_log_on_epoch)
            self.log("val/adv_loss", adv_loss, sync_dist=self.train_log_on_epoch)

class ConDomainClassifier(nn.Module):
    def __init__(self, in_feat, out_class, hidden ):
        super(ConDomainClassifier, self).__init__()
        
        self.classfier = MLP(in_feat, out_class, hidden)
        self.loss = torch.nn.CrossEntropyLoss()
        self.num_domian_labels = 3

    def forward(self, feature,y,date,coeff=1):
        loss = [0 for i in range(self.num_domian_labels)]
        for i in range(self.num_domian_labels):
            idx = date==i
            loss+=self.loss(feature[idx],y[idx])
    
        return loss
    