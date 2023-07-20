from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import MMD_loss
from models.loss_module.domain_adv_loss import DALoss


class MMD_AAE(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization=False
        if self.config['experiment']['recontruct_coeff']:
            self.decoder =  Decoder(256, 2, self.config['model']['linear'])
            self.reconstruct_criterion= nn.MSELoss()
        
        if self.config['experiment']['adv_coeff']:
            self.discriminator = Discriminator(in_feature=512*8, 
                                            hidden_units_size=config['model']['hidden_units_size']
                                            )
            self.adv_criterion = DALoss()
        if self.config['experiment']['MMD_coeff']:
            self.mmd_loss = MMD_loss(MMD_sample_size=config['experiment']['MMD_sample_size'])
        
    def training_step(self, batch, batch_idx):
        optimizers_retriver = self.give_opt_and_sch()
        fe_opt, fe_sch = next(optimizers_retriver)
        
        
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False,on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step= self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        
        if self.config['experiment']['MMD_coeff']:
            mmd_opt, mmd_sch = next(optimizers_retriver)
            idx1, idx2, idx3 = date==0, date==1, date==2
            
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            mmd1, mmd2, mmd3 = self.mmd_loss(feat1,feat1),self.mmd_loss(feat1,feat3),self.mmd_loss(feat2,feat3),
            loss +=  self.config['experiment']['MMD_coeff']*(mmd1+mmd2+mmd3)
            self.log("train/mmd", mmd1+mmd2+mmd3, prog_bar=True, on_step= self.train_log_on_epoch,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
        
            mmd_opt.zero_grad()
            self.manual_backward(loss,
                    retain_graph= self.config['experiment']['recontruct_coeff']>0 or self.config['experiment']['adv_coeff']>0
                    )
            mmd_opt.step()
            # mmd_sch.step()s
        
        # fe_opt.zero_grad()
        # # self.manual_backward(loss, retain_graph= True)
        # self.manual_backward(mmd1+mmd2+mmd3)
        # self.log("train/total_loss",loss, prog_bar=True, on_step= self.train_log_on_epoch,
        #              on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
        
        # fe_opt.step()
        # fe_sch.step()
        
        
        
        
        
        
        
        
        
        if self.config['experiment']['recontruct_coeff']:
            recons_opt, recons_sch = next(optimizers_retriver)
            decoded_original_signal = self.decoder(feature)
            reconstruct_loss = self.reconstruct_criterion(decoded_original_signal,x)
            loss += self.config['experiment']['recontruct_coeff']*reconstruct_loss 
            self.log("train/recons_loss", reconstruct_loss, on_step=not self.train_log_on_epoch,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 ) 
            
            recons_opt.zero_grad()
            self.manual_backward(reconstruct_loss,retain_graph=self.config['experiment']['adv_coeff']>0)
            recons_opt.step()
            recons_sch.step()
            
        
        if self.config['experiment']['adv_coeff']:
            adv_opt,adv_sch = next(optimizers_retriver)
            rgl_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["rgl_kick_in_position"]//self.config['dataset']["batch_size"])
            loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["loss_kick_in_position"]//self.config['dataset']["batch_size"])
            feat = feature.view(len(feature), -1)
            prior_feat = torch.tensor(np.random.laplace(loc=0,scale=0.1,size=feat.shape)).float()
            prior_feat = prior_feat.to(feat)
            feature_together = torch.cat([feat, prior_feat], dim=0)
            # label : real_feat->1, prior_feat->0
            domain_prediction = self.discriminator(feature_together, rgl_coeff)
            adv_loss = self.adv_criterion(domain_prediction, loss_coeff)
            loss += self.config['experiment']['adv_coeff']*adv_loss
            self.log("train/domain_prediction_feat", torch.mean(domain_prediction[0:feat.shape[0]]),
                     on_step=not self.train_log_on_epoch, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/domain_prediction_prior", torch.mean(domain_prediction[feat.shape[0]:]), 
                     on_step=not self.train_log_on_epoch, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/adv_loss", adv_loss, on_step=not self.train_log_on_epoch,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)

            adv_opt.zero_grad()
            self.manual_backward(adv_loss)
            adv_opt.step()
            adv_sch.step()
        
        # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 50 == 0:
        #     sch_retriver = self.give_sch()       
        #     fe_sch = next(sch_retriver)             
        #     if self.config['experiment']['recontruct_coeff']:
        #         recons_sch = next(sch_retriver)        
        #         recons_sch.step(self.trainer.callback_metrics["train/recons_loss"])
        #     if self.config['experiment']['adv_coeff']:
        #         adv_sch = next(sch_retriver)   
        #         adv_sch.step()    
        # return loss
     
    def configure_optimizers(self):
     
        # print(self.parameters())
        lr = self.config['feature_extractor']['learning_rate']
        fe_optimizer = torch.optim.SGD(self.encoder.parameters(),
                                        lr =lr,
                                        momentum=self.config['feature_extractor']['momentum'],
                                        weight_decay=self.config['feature_extractor']['weight_decay'], 
                                        nesterov=self.config['feature_extractor']['nesterov']=="True")
        fe_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fe_optimizer, mode='min', factor=0.5, patience=3)
        # fe_scheduler = torch.optim.lr_scheduler.CyclicLR(fe_optimizer,base_lr=lr/100, max_lr=lr*2,step_size_up=2000)
        optimizer_list = [fe_optimizer]
        scheduler_list = [fe_scheduler]
        
        if self.config['experiment']['MMD_coeff']:
            lr = self.config['MMD']['learning_rate']
            MMD_optimizer = torch.optim.Adam(self.encoder.parameters(),
                                            lr = lr,
                                            weight_decay=self.config['MMD']['weight_decay'],
                                            )     
            MMD_scheduler = torch.optim.lr_scheduler.CyclicLR(fe_optimizer,base_lr=lr/100, max_lr=lr*2,step_size_up=1850)
            optimizer_list.append(MMD_optimizer)
            scheduler_list.append(MMD_scheduler)
            
        if self.config['experiment']['recontruct_coeff']:
            lr =self.config['decoder']['learning_rate']
            recons_optimizer = torch.optim.SGD(self.decoder.parameters(),
                                            lr = lr,
                                            momentum=self.config['decoder']['momentum'],
                                            weight_decay=self.config['decoder']['weight_decay'], 
                                            nesterov=self.config['decoder']['nesterov']=="True")
            recons_scheduler = torch.optim.lr_scheduler.CyclicLR(fe_optimizer,base_lr=lr/100, max_lr=lr*2,step_size_up=2200)
            optimizer_list.append(recons_optimizer)
            scheduler_list.append(recons_scheduler)
            
        if self.config['experiment']['adv_coeff']:
            lr = self.config['adv_classifier']['learning_rate']
            adv_optimizer = torch.optim.SGD(self.discriminator.parameters(),
                                            lr = lr,
                                            momentum=self.config['adv_classifier']['momentum'],
                                            weight_decay=self.config['adv_classifier']['weight_decay'], 
                                            nesterov=self.config['adv_classifier']['nesterov']=="True")
            adv_scheduler = torch.optim.lr_scheduler.CyclicLR(adv_optimizer,base_lr=lr/100, max_lr=lr*2,step_size_up=1700)
            optimizer_list.append(adv_optimizer)
            scheduler_list.append(adv_scheduler)
        return optimizer_list,scheduler_list
    
    def on_train_epoch_end(self):   
        sch_retriver = self.give_sch()       
        fe_sch = next(sch_retriver)
        fe_sch.step(self.trainer.callback_metrics["val/loss"])
        
        # if self.config['experiment']['MMD_coeff']:
        #     MMD_sch = next(sch_retriver)        
        #     MMD_sch.step(self.trainer.callback_metrics["train/mmd"])
        # if self.config['experiment']['recontruct_coeff']:
        #     recons_sch = next(sch_retriver)        
        #     recons_sch.step(self.trainer.callback_metrics["train/recons_loss"])
        # if self.config['experiment']['adv_coeff']:
        #     adv_sch = next(sch_retriver)   
        #     adv_sch.step()

    
'''additional debug step
1. removed schs 
2. progress on step
3. use one opt, recover schs

'''