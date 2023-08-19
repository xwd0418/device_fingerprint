from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import MMD_loss
from models.loss_module.JSD import JSD
import math
from models.loss_module.utils import compute_discrepency


class ConDG(Baseline_Resnet):
    def __init__(self, config,datamodule):
        super().__init__(config)
        # self.automatic_optimization = False

        
        self.num_classes = 5 if config['dataset'].get('img') else 150 
        self.num_domains = 3 if self.config['dataset'].get('old_split') or config['dataset'].get('img') else 2
        
        self.on_step = False
        self.last_batch_general_domain_accu = 0.0
        self.general_domain_accu = Accuracy( task="multiclass", num_classes=self.num_domains)
        if config['experiment']['discrepency_metric'] == "JSD":    
            self.discrepency_loss_func = JSD()
        # self.set_adv_coeffs()
        elif config['experiment']['discrepency_metric'] == "MMD":
            self.discrepency_loss_func = MMD_loss(**config['experiment']['MMD_kwargs'])
        
        self.label_distribution = datamodule.label_distribution
     
        adv_in_feat_size = 2048 if  self.config['dataset'].get('img') else 512*8
        if self.config['experiment']['con_domain_coeff']:
            for i in range(self.num_classes):
                setattr(self, f'conditional_classifier_{i}' , 
                                    AdvMLPClassifier(
                                        in_feat_size=adv_in_feat_size, out_class=self.num_domains, 
                                        hidden_units_size=self.config['model']['con_hidden_units_size'],
                                        num_classes = self.num_classes,
                                        label_distribution=self.label_distribution,
                                        class_conditional=True
                                    ) 
                )
                setattr(self,f"last_batch_con_domain_accu_{i}" , 0.0)
                setattr(self, f"con_accuracy_{i}",  Accuracy( task="multiclass", num_classes=self.num_domains))
        if self.config['experiment']['weighted_domain_coeff'] :        
            self.general_domain_classifier = AdvMLPClassifier(
                                in_feat_size=adv_in_feat_size, out_class=self.num_domains,
                                hidden_units_size=self.config['model']['general_hidden_units_size'], 
                                num_classes = self.num_classes,
                                label_distribution=self.label_distribution,
                                class_conditional = False
                                ) # using default reductioon mean 
            self.single_domain_accus = [Accuracy( task="multiclass", num_classes=self.num_domains) for _ in range(self.num_domains)]
        # a 2d array of #{domains} x #{labels} 
        
    def training_step(self, batch, batch_idx):
        self.set_adv_coeffs()
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        cls_loss = self.criterion(logits, y)
        loss = cls_loss
        
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False, on_epoch=self.train_log_on_epoch, on_step=self.on_step, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_epoch=self.train_log_on_epoch, on_step=self.on_step, sync_dist=torch.cuda.device_count()>1)

        
        if self.config['dataset'].get('img'):
            feature = self.encoder.pooled_feature.view(len(feature), -1)
        # conditional domain classifier
        if self.config['experiment']['con_domain_coeff']: 
            con_domain_loss = 0           
            for class_idx in range(self.num_classes):
                idx = y==class_idx
                if torch.any(idx):
                    class_feat, class_y, class_date = feature[idx], y[idx], date[idx]
                    
                    curr_classifier = getattr(self, f'conditional_classifier_{class_idx}')
                    curr_con_coeff = getattr(self, f'con_rgl_coeff_{class_idx}')
                    con_pred_logits, single_class_loss = curr_classifier(class_feat, class_y, class_date,
                                                hook_coeff=curr_con_coeff, loss_coeff=self.con_loss_coeff) 
                    con_domain_loss += single_class_loss
                    curr_metric = getattr(self, f"con_accuracy_{class_idx}" )
                    setattr( self,f"last_batch_con_domain_accu_{class_idx}",  
                            curr_metric(torch.argmax(con_pred_logits, dim=1), date[idx]) 
                            )
            con_domain_loss /= self.num_classes
                
            loss +=  self.config['experiment']['con_domain_coeff'] * con_domain_loss
            self.log("train/con_domain_loss", con_domain_loss, on_step=self.on_step,
                        on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
            avg_con_accuracy = sum([getattr(self,f"last_batch_con_domain_accu_{i}") for i in range (self.num_classes)])/self.num_classes
            
            self.log("train/con_domain_acc", avg_con_accuracy, prog_bar=True, on_step=self.on_step, 
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)

            avg_hook_coeff = sum([getattr(self,f'con_rgl_coeff_{i}') for i in range (self.num_classes)])/self.num_classes
            self.log("train/hook_coeff", avg_hook_coeff, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
            self.log("train/loss_coeff", self.con_loss_coeff, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )







        if self.config['experiment']['weighted_domain_coeff']:   
            # weighted domain classifier
            
            # weighted_pred_logits , weighted_domain_loss = self.general_domain_classifier(feature, y, date, hook_coeff=self.general_rgl_coeff, loss_coeff=self.general_loss_coeff) 
            # loss += self.config['experiment']['weighted_domain_coeff'] * weighted_domain_loss

            # self.last_batch_general_domain_accu = self.general_domain_accu(torch.argmax(weighted_pred_logits, dim=1), date)
            self.log("train/weighted_domain_accu", self.last_batch_general_domain_accu, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

            general_domain_loss = 0
            self.last_batch_general_domain_accu = 0
            for i in range(self.num_domains):
                date_idx = date==i
                weighted_pred_logits , weighted_domain_loss = self.general_domain_classifier(feature[date_idx], y[date_idx], date[date_idx],
                                                                        hook_coeff = self.general_rgl_coeff, loss_coeff=self.general_loss_coeff
                                                                        # hook_coeff = 0.5, loss_coeff = 1.0
                                                                                             ) 
                accu = self.single_domain_accus[i].to(feature)(torch.argmax(weighted_pred_logits, dim=1), date[date_idx])
                general_domain_loss += weighted_domain_loss
                self.last_batch_general_domain_accu += accu
                self.log(f"train/weighted_domain_{i}_accu", accu, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )
            
            loss += self.config['experiment']['weighted_domain_coeff'] * general_domain_loss
            self.last_batch_general_domain_accu /= self.num_domains
            self.log("train/general_rgl_coeff", self.general_rgl_coeff, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

            self.log("train/general_loss_coeff", self.general_loss_coeff, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

            





        discrepency_loss = None
        if self.config['experiment'].get('discrepency_metric') :
            discrepency_loss = compute_discrepency(self.discrepency_loss_func, feature, date, len(batch))
        if  discrepency_loss is not None:   
            loss +=  self.config['experiment']['discrepency_coeff']*discrepency_loss
            self.log("train/discrepency_loss", discrepency_loss, on_step=self.on_step,
                        on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )


        self.log("train/loss", loss, on_step=self.on_step,
                     on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1 )

        return  loss

    def set_adv_coeffs(self):
        if self.config['experiment']['con_domain_coeff']:
            self.con_loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["con_loss_kick_in_position"]//self.config['dataset']["batch_size"])
            for i in range(self.num_classes):
                    accu = min(getattr(self,f"last_batch_con_domain_accu_{i}"), 0.99999)
                    setattr(self, f'con_rgl_coeff_{i}', math.log(accu/(1-accu)*(self.num_domains-1) +1, 2)) 
                
                
        if self.config['experiment']['weighted_domain_coeff']:         
            self.general_loss_coeff = self.calc_coeff(self.global_step, kick_in_iter = self.config["experiment"]["general_loss_kick_in_position"]//self.config['dataset']["batch_size"])
            accu = min(self.last_batch_general_domain_accu, 0.99999)
            self.general_rgl_coeff = math.log(accu/(1-accu)*(self.num_domains-1) +1, 2)
        
        # pass
        """
        torch.

        when accu is 0, it is log(1)=0,
        when accu is 1, it is log(large), so a value close to 19,
        when accu is 1/num_domain, which is purely guessing, accu/(1-accu) =1/(domain-1) 
        """