from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import MMD_loss
from models.loss_module.JSD import JSD


class ConDG(Baseline_Resnet):
    def __init__(self, config,datamodule):
        super().__init__(config)
        self.num_classes = 150
        self.num_domains = 4-1
        self.JSD = JSD()
        self.setup_adv_coeffs()
        if config['experiment']['discrepency_metric'] == "MMD":
            self.mmd_loss = MMD_loss(MMD_sample_size=config['experiment']['MMD_sample_size'])

        
        self.label_distribution = datamodule.label_distribution
        # self.condition_domain_classifiers =  [ AdvMLPClassifier(
        #                             in_feat_size=512*8, out_class=self.num_domains, 
        #                             hidden_units_size=self.config['model']['con_hidden_units_size'],
        #                             label_distribution=self.label_distribution,
        #                             class_conditional=True
        #                         ) for i in range(self.num_classes) ]
        if self.config['experiment']['con_domain_coeff']:
            for i in range(self.num_classes):
                setattr(self, f'conditional_classifier_{i}' , 
                                    AdvMLPClassifier(
                                        in_feat_size=512*8, out_class=self.num_domains, 
                                        hidden_units_size=self.config['model']['con_hidden_units_size'],
                                        label_distribution=self.label_distribution,
                                        class_conditional=True
                                    ) 
                )
        if self.config['experiment']['weighted_domain_coeff']:        
            self.general_domain_classifier = AdvMLPClassifier(
                                in_feat_size=512*8, out_class=self.num_domains,
                                hidden_units_size=self.config['model']['general_hidden_units_size'], 
                                label_distribution=self.label_distribution,
                                class_conditional = False
                                ) # using default reductioon mean 
        
        # a 2d array of #{domains} x #{labels} 
        
    def training_step(self, batch, batch_idx):
        self.setup_adv_coeffs()
        
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)

        # conditional domain classifier
        if self.config['experiment']['con_domain_coeff']: 
            con_domain_loss = 0           
            for class_idx in range(self.num_classes):
                idx = y==class_idx
                if torch.any(idx):
                    class_feat, class_y, class_date = feature[idx], y[idx], date[idx]
                    con_domain_loss += getattr(self, f'conditional_classifier_{class_idx}')(class_feat, class_y, class_date,
                                                hook_coeff=self.con_rgl_coeff, loss_coeff=self.con_loss_coeff) 
            loss +=  self.config['experiment']['con_domain_coeff'] * con_domain_loss
            self.log("train/con_domain_loss", con_domain_loss, on_step=not self.train_log_on_epoch,
                        on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

        if self.config['experiment']['weighted_domain_coeff']:   
            # weighted domain classifier
            weighted_domain_loss = self.general_domain_classifier(feature, y, date, hook_coeff=self.general_rgl_coeff, loss_coeff=self.general_loss_coeff) 
            loss += self.config['experiment']['weighted_domain_coeff'] * weighted_domain_loss
            self.log("train/weighted_domain_loss", weighted_domain_loss, on_step=not self.train_log_on_epoch,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

        discrepency_loss = None
        if self.config['experiment'].get('discrepency_metric') == "MMD":
            idx1, idx2, idx3 = date==0, date==1, date==2            
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            mmd1, mmd2, mmd3 = self.mmd_loss(feat1,feat1),self.mmd_loss(feat1,feat3),self.mmd_loss(feat2,feat3),
            discrepency_loss = mmd1+mmd2+mmd3
            
        if self.config['experiment'].get('discrepency_metric')== "JSD":
            idx1, idx2, idx3 = date==0, date==1, date==2  
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            JSD1, JSD2, JSD3 = self.JSD(feat1,feat1),self.JSD(feat1,feat3),self.JSD(feat2,feat3),
            discrepency_loss = JSD1+JSD2+JSD3
        
        if  discrepency_loss is not None:   
            loss +=  self.config['experiment']['discrepency_coeff']*discrepency_loss
            self.log("train/discrepency_loss", discrepency_loss, on_step=not self.train_log_on_epoch,
                        on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )


        self.log("train/loss", loss, on_step=not self.train_log_on_epoch,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

        return loss

    def setup_adv_coeffs(self):
        if self.config['experiment']['con_domain_coeff']:
            self.con_rgl_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["con_rgl_kick_in_position"]//self.config['dataset']["batch_size"])
            self.con_loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["con_loss_kick_in_position"]//self.config['dataset']["batch_size"])
        
        if self.config['experiment']['weighted_domain_coeff']:        
            self.general_rgl_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["general_rgl_kick_in_position"]//self.config['dataset']["batch_size"])
            self.general_loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["general_loss_kick_in_position"]//self.config['dataset']["batch_size"])
     