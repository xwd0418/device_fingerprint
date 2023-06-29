import torch
import  torch.nn as nn
class DALoss(nn.Module):
    ''' Ref: https://github.com/thuml/CDAN/blob/master/pytorch/loss.py
    '''

    def __init__(self):
        super(DALoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, ad_out, coeff=1.0, dc_target=None):
        batch_size = ad_out.shape[0]//2
        if dc_target == None:
           dc_target = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)), 0).float()
        loss = self.criterion(ad_out.view(-1), dc_target.view(-1))
        # after_sig = nn.Sigmoid()(ad_out).squeeze()
        # loss = nn.BCELoss(reduction='none')(after_sig, dc_target)
        # print("my computed daloss is ",loss)
        
        return coeff*torch.mean(loss.squeeze())