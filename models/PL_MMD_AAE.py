from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import mmd
from models.loss_module.domain_adv_loss import DALoss


class MMD_AAE(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.decoder =  Decoder(256, 2, self.config['model']['linear'])
        self.discriminator = Discriminator(in_feature=512*8, 
                                           hidden_size=self.config['model']['adv_hidden_size']
                                           )
        self.adv_criterion = DALoss()
        self.reconstruct_criterion= nn.MSELoss()
        
    def training_step(self, batch, batch_idx):
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)   

        if self.config['experiment']['recontruct_coeff']:
            
            decoded_original_signal = self.decoder(feature)
            reconstruct_loss = self.reconstruct_criterion(decoded_original_signal,x)
            loss += self.config['experiment']['recontruct_coeff']*reconstruct_loss 
            self.log("train/recons_loss", reconstruct_loss, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch ) 
        
        if self.config['experiment']['MMD_coeff']:
            idx1, idx2, idx3 = date==0, date==1, date==2
            
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            mmd1, mmd2, mmd3 = mmd(feat1,feat1),mmd(feat1,feat3),mmd(feat2,feat3),
            loss +=  self.config['experiment']['MMD_coeff']*(mmd1+mmd2+mmd3)
            self.log("train/mmd", mmd1+mmd2+mmd3, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch )
        
        if self.config['experiment']['adv_coeff']:
            coeff = self.calc_coeff(self.global_step)
            feat = feature.view(len(feature), -1)
            prior_feat = torch.tensor(np.random.laplace(loc=0,scale=0.1,size=feat.shape)).float()
            prior_feat = prior_feat.to(feat)
            feature_together = torch.cat([feat, prior_feat], dim=0)
            domain_prediction = self.discriminator(feature_together, coeff)
            adv_loss = self.adv_criterion(domain_prediction,coeff)
            loss += self.config['experiment']['adv_coeff']*adv_loss
            self.log("train/domain_prediction_feat", torch.mean(domain_prediction[0:feat.shape[0]]),on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)
            self.log("train/domain_prediction_prior", torch.mean(domain_prediction[feat.shape[0]:]), on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)
            self.log("train/adv_loss", adv_loss, on_epoch=self.train_log_on_epoch, sync_dist=self.train_log_on_epoch)

            
        return loss
     
    def validation_step(self, batch, batch_idx):
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val/cls_loss", loss,  prog_bar=False, sync_dist=self.train_log_on_epoch)
        self.log("val/acc", self.val_accuracy, prog_bar=True, sync_dist=self.train_log_on_epoch)   

        if self.config['experiment']['recontruct_coeff']:
            
            decoded_original_signal = self.decoder(feature)
            reconstruct_loss = self.reconstruct_criterion(decoded_original_signal,x)
            loss += self.config['experiment']['recontruct_coeff']*reconstruct_loss 
            self.log("val/recons_loss", reconstruct_loss, sync_dist=self.train_log_on_epoch) 
        
        if self.config['experiment']['MMD_coeff']:
            idx1, idx2, idx3 = date==0, date==1, date==2
            
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            mmd1, mmd2, mmd3 = mmd(feat1,feat1),mmd(feat1,feat3),mmd(feat2,feat3),
            loss +=  self.config['experiment']['MMD_coeff']*(mmd1+mmd2+mmd3)
            self.log("val/mmd", mmd1+mmd2+mmd3, sync_dist=self.train_log_on_epoch)
        
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

    