from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import MMD_loss
from models.loss_module.domain_adv_loss import DALoss
from torchmetrics import Accuracy
from models.loss_module.utils import compute_discrepency

class MMD_AAE(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.feat_accuracy = Accuracy( task = 'binary')
        self.prior_accuracy = Accuracy( task = 'binary')
        self.on_step = False
        self.last_batch_domain_accu = 0.0

        
        self.automatic_optimization = self.config['experiment'].get('single_optimizer') is not False
        if self.config['experiment']['recontruct_coeff']:
            in_channel, out_channel = (2048, 3) if self.config['dataset'].get('img') else (512,2)
            self.decoder =  Decoder(in_channel, out_channel, self.config['model']['linear'],
                                    twoD=config['dataset'].get('img'))
            self.reconstruct_criterion= nn.MSELoss()
        
        if self.config['experiment']['adv_coeff']:
            if not config['model'].get('hidden_units_size'):
                config['model']['hidden_units_size'] = [ config['model']['adv_hidden_size_range'] ]
            in_feature = 2048 if self.config['dataset'].get('img') else 512*8
            self.discriminator = Discriminator(in_feature=in_feature, 
                                            hidden_units_size=config['model']['hidden_units_size']
                                            )
            self.adv_criterion = DALoss()
        if self.config['experiment']['MMD_coeff']:            
                self.mmd_loss = MMD_loss(**config['experiment']['MMD_kwargs'])
        
        
        self.load_pretrained()
            
        # '''just fot this time'''
        # model_path = '/root/exps_dev/MMD_AAE/only_adv/version_0/checkpoints/last.ckpt'
        # checkpoint = torch.load(model_path)
        # print('successfully loaded discriminator')
        # self.load_state_dict(checkpoint['state_dict'])
        
        # for param in self.discriminator.parameters():
        #     param.requires_grad = False
            
            
            
            
            
    def training_step(self, batch, batch_idx):
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False,on_step = self.on_step,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step = self.on_step,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   
           
        if not self.automatic_optimization:
            optimizers_retriver = self.give_opt_and_sch()
            fe_opt, fe_sch = next(optimizers_retriver)
            fe_opt.zero_grad()
            self.manual_backward(loss, retain_graph= True)
            # self.log("train/total_loss",loss, prog_bar=True, on_step = self.on_step,
            #              on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
            
            fe_opt.step()
            fe_sch.step()
        
        
        
        
        
        
        if self.config['experiment']['MMD_coeff']:
            total_mmd = compute_discrepency(self.mmd_loss, feature, date, len(batch))
            loss +=  self.config['experiment']['MMD_coeff']*(total_mmd)
            self.log("train/mmd", total_mmd, prog_bar=True, on_step = self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
        
            if not self.automatic_optimization:
                mmd_opt, mmd_sch = next(optimizers_retriver)
                mmd_opt.zero_grad()
                self.manual_backward(total_mmd,
                        retain_graph= self.config['experiment']['recontruct_coeff']>0 or self.config['experiment']['adv_coeff']>0
                        )
                mmd_opt.step()
                mmd_sch.step()
        
        
        
        
        
        
        
        
        
        
        if self.config['experiment']['recontruct_coeff']:
            decoded_original_signal = self.decoder(feature)
            reconstruct_loss = self.reconstruct_criterion(decoded_original_signal,x)
            loss += self.config['experiment']['recontruct_coeff']*reconstruct_loss 
            self.log("train/recons_loss", reconstruct_loss, on_step = self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 ) 
            
            if not self.automatic_optimization:
                recons_opt, recons_sch = next(optimizers_retriver)
                recons_opt.zero_grad()
                self.manual_backward(reconstruct_loss,retain_graph=self.config['experiment']['adv_coeff']>0)
                recons_opt.step()
                recons_sch.step()
     
     
     
     
     
            
        
        if self.config['experiment']['adv_coeff']:
            loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["loss_kick_in_position"]//self.config['dataset']["batch_size"])
            # rgl_coeff = accu/(1-accu)
            max_rgl_coeff = self.config['experiment'].get('max_rgl_coeff')
            if max_rgl_coeff is None:
                max_rgl_coeff = 5
            rgl_coeff = max_rgl_coeff ** self.last_batch_domain_accu
            """
            10 is a hand-picked number. It means when accuracy is 10%, the coeff will be 1, 
            which means that accuracy will bs 
            """

            if self.config['dataset'].get('img'):
                feat = self.encoder.pooled_feature.view(len(feature), -1)
            else:
                feat = feature.view(len(feature), -1)
            prior_feat = self.get_prior_feat(shape = feat.shape)
            prior_feat = prior_feat.to(feat)
            feature_together = torch.cat([feat, prior_feat], dim=0)
            # label : real_feat->1, prior_feat->0
            domain_prediction = self.discriminator(feature_together, rgl_coeff)
            self.last_batch_domain_accu = self.feat_accuracy(torch.nn.Sigmoid()(domain_prediction.view(-1)[0:len(feat)]), torch.ones(len(feat)).to(feat))
            self.prior_accuracy(torch.nn.Sigmoid()(domain_prediction.view(-1)[len(feat):]), torch.zeros(len(feat)).to(feat))
            adv_loss = self.adv_criterion(domain_prediction, loss_coeff)
            loss += self.config['experiment']['adv_coeff']*adv_loss
           
            self.log("train/domain_feat_pred", torch.mean(domain_prediction[0:feat.shape[0]]),
                     on_step = self.on_step, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/domain_prior_pred", torch.mean(domain_prediction[feat.shape[0]:]), 
                     on_step = self.on_step, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/adv_loss", adv_loss, on_step = self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/domain_accu", self.feat_accuracy, on_step = self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            # self.log("train/prior_accu", self.prior_accuracy, on_step = self.on_step,
            #          on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            self.log("train/hook_coeff", rgl_coeff, on_step = self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
            
            if not self.automatic_optimization:
                adv_opt,adv_sch = next(optimizers_retriver)
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
        
        # self.log("train/total_loss", loss, prog_bar=True, on_step = self.on_step,
        #              on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )  

        return loss
    
    def get_prior_feat(self, shape):
        distribution_type = self.config['experiment'].get('prior_distribution') 
        if distribution_type is None or distribution_type == "Laplace":
            return  torch.tensor(np.random.laplace(loc=0,scale=0.1,size=shape)).float()
        if distribution_type == "Normal":
            return  torch.tensor(np.random.normal(loc=0,scale=0.1,size=shape)).float()
        
    def configure_optimizers(self):
        if self.automatic_optimization:
            return super().configure_optimizers()
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
        super().on_train_epoch_end()
        if not self.automatic_optimization:
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

    def load_pretrained(self):
        ckpt = self.config['experiment'].get('load_from_MMD_AAE')
        if ckpt :
            ckpt = torch.load(ckpt)
            # print(f'loaded from {ckpt}')
            self.load_state_dict(ckpt['state_dict'],strict=False) 
            
'''additional debug step
1. removed schs 
2. progress on step
3. use one opt, recover schs


'''