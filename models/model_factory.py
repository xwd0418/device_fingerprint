import torch
import torch.nn as nn, numpy as np
import torch.nn.functional as F 
from typing import Callable, List, Optional, Sequence, Tuple, Union
# from torchvision.ops import MLP 
        
'''decoder!'''     
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, oneD=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        conv1  = nn.Conv1d if oneD else nn.Conv2d
        bn1 = nn.BatchNorm1d if oneD else nn.BatchNorm2d
        conv2 = nn.Conv1d if oneD else nn.Conv2d
        bn2 = nn.BatchNorm1d if oneD else nn.BatchNorm2d
        self.double_conv = nn.Sequential(
            conv1(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            bn1(mid_channels),
            nn.ReLU(inplace=True),
            conv2(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            bn2(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, linear=True, oneD=True):
        super().__init__()
        self.oneD =oneD

        # if bilinear, use the normal convolutions to reduce the number of channels
        if linear:
            mode='linear' if oneD else "bilinear"
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, oneD=oneD) 
                
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) if not oneD else \
                 nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, oneD=oneD)

    def forward(self, x1):
        expect_length = x1.size()[2]*2
        x1 = self.up(x1)
        # input is CHW
        
        diff = expect_length-x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, oneD=True):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if not oneD else \
            nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, linear = True, twoD=False):
        oneD = twoD is None or not twoD 
        super(Decoder, self).__init__()
        self.linear = linear
        
        self.up1 = Up(n_channels_in, 256 , linear, oneD=oneD)
        self.up2 = Up(256, 128 , linear, oneD=oneD)
        self.up3 = Up(128, 64, linear, oneD=oneD)
        self.up4 = Up(64, 32, linear, oneD=oneD)
        self.up5 = Up(32, 16, linear, oneD=oneD)
        self.outc = OutConv(16, n_channels_out, oneD=oneD)

    def forward(self, x):
        # print("why is 2",x.shape
        # print('x5 shape is ', x5.shape)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        reconstruction = self.outc(x)

        return reconstruction
    
'''discriminator!'''
class Discriminator(nn.Module):
  def __init__(self, in_feature, hidden_units_size):
    super(Discriminator, self).__init__()
    self.mlp = MLP(in_feature, hidden_units_size+[1], dropout=0.5)

  def forward(self, x, hook_coeff):
    # x = x * 1.0
    # print("self.training,\n\n", self.training)
    if self.training:
        x.register_hook(grl_hook(hook_coeff))
        # pass
    y = self.mlp(x)
    # y = self.sigmoid(y)
    return y

class AdvMLPClassifier(nn.Module):
    def __init__(self, in_feat_size, out_class, hidden_units_size, num_classes,
                 CE_reduction='none', label_distribution=None, class_conditional=True,):
        super(AdvMLPClassifier, self).__init__()
        self.classfier = MLP(in_feat_size, hidden_units_size+[out_class])
        self.loss = torch.nn.CrossEntropyLoss(reduction=CE_reduction)
        self.class_conditional = class_conditional
        self.L = num_classes
        self.label_distribution = label_distribution
        self.domain_distribution = torch.sum(label_distribution, dim=1)
        self.total_size = torch.sum(self.domain_distribution)
        assert(torch.sum(label_distribution[0])==self.domain_distribution[0])
        
    def forward(self, feature, y, date, hook_coeff, loss_coeff ):
        feature = feature.view(len(feature),-1)
        if self.training:
            feature.register_hook(grl_hook(hook_coeff))
    
        prediction = self.classfier(feature)
        loss = self.loss(prediction,date)
        
        # move to cuda
        self.label_distribution = self.label_distribution.to(feature)
        self.total_size = self.total_size.to(feature)
        self.domain_distribution = self.domain_distribution.to(feature)
        
        if self.class_conditional:
            # loss = loss * self.domain_distribution[date] / self.L / self.label_distribution[date,y]
            # loss = loss * self.total_size / self.domain_distribution[date] 
            #cancelling out terms above
            loss = loss * self.total_size / self.L / self.label_distribution[date,y]
        else:
            loss = loss * self.total_size / self.domain_distribution[date]
            
        return prediction, loss.mean()*loss_coeff/(hook_coeff+1)
        # divided by hook_coeff+1 is because we want to have the sum(gradient coeffs of the two parts)==1
        # hook_coeff can be pretty large (range from 0.001 to 19) 
     
def grl_hook(hook_coeff):
    def fun1(grad):
        return -hook_coeff*grad.clone()
    return fun1

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = False,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1] :
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        # layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)
        
if __name__ == "__main__":
    dis = Discriminator(4,[64])
    input = torch.tensor([54,45,65,76])
    target = []