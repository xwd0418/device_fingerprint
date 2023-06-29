from models.PL_resnet import *
from models.model_factory import *
from models.loss_module.MMD import mmd
from models.loss_module.domain_adv_loss import DALoss


class MMD_AAE(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.decoder =  Decoder(256, 2, self.config['model']['linear'])
        self.discriminator = Discriminator(in_feature=512, 
                                           hidden_size=self.config['model']['adv_hidden_size']
                                           )
        self.adv_criterion = DALoss()
        
    def training_step(self, batch, batch_idx):
        x, y, date = self.unpack_batch(batch, need_date=True)
        logits, feature = self(x, feat=True)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/cls_loss", loss,  prog_bar=False)
        self.log("train/acc", self.train_accuracy, prog_bar=True)   

        if self.config['experiment']['recontruct_coeff']:
            
            decoded_original_signal = self.decoder(feature)
            reconstruct_loss = nn.MSELoss(decoded_original_signal,x)
            loss += self.config['experiment']['recontruct_coeff']*reconstruct_loss  
        
        if self.config['experiment']['MME_coeff']:
            idx1, idx2, idx3 = date==0, date==1, date==2
            
            feat1, feat2, feat3 = feature[idx1], feature[idx2], feature[idx3]
            feat1, feat2, feat3 = feat1.view(len(feat1), -1), feat2.view(len(feat2), -1), feat3.view(len(feat3), -1)
            assert (len(feat1)+len(feat2)+len(feat3)==len(feature))
            mmd1, mmd2, mmd3 = mmd(feat1,feat1),mmd(feat1,feat3),mmd(feat2,feat3),
            loss +=  self.config['experiment']['MMD_coeff']*(mmd1+mmd2+mmd3)
        
        if self.config['experiment']['adv_coeff']:
            coeff = self.calc_coeff(self.global_step)
            feat = feature.view(len(feature), -1)
            prior_feat = torch.tensor(np.random.laplace(loc=0,scale=0.1,size=feat.shape))
            feature_together = torch.cat([feat, prior_feat], dim=0)
            domain_prediction = self.discriminator(feature_together, coeff)
            adv_loss = self.adv_criterion(domain_prediction,coeff)
            loss += self.config['experiment']['adv_coeff']*adv_loss
            
        return loss
     
    def unpack_batch(batch, need_date=False):
        if need_date:
            x,y,date = batch    
            return x,y,date
        else:
            x,y = batch
            return x,y
    def get_all(self) :
        pickleFile = open("/root/device_fingerprint/dataset/ManyTx.pkl","rb")
        all_info = pickle.load(pickleFile)
        data = all_info['data']

        source_data,target_data = [],[]
        for label in range(len(data)):
            # one_hot_encoded = F.one_hot(torch.tensor([label]), num_classes=len(data))
            for i in data[label]:
                for date, j in enumerate(i[0:3]):
                    for k in j[1]:
                        source_data.append((k.T.astype("float32"),label, date)) 
                for j in [ i[3] ]: # this seems dumb, just for sake of pretty alignment
                    for k in j[1]: # delete this line if we need to put the 50 together
                        target_data.append((k.T.astype("float32"),label,3)) 
                        # shape of k is 256 * 2
        return source_data,target_data 

    