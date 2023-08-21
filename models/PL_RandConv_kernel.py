"""
implementing random conv from
https://arxiv.org/pdf/2007.13003.pdf
"""

from models.PL_resnet import *
from models.model_factory import *
from torch.nn import Conv1d
import math


class RandConv_kernel(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
         
        self.consistency_loss_coeff = config['experiment']['consistency_loss_coeff']
        # if self.consistency_loss_coeff:
        #     self.consistency_loss_func = nn.KLDivLoss(reduction="batchmean")
          
        kernel_sizes = [i*2+3 for i in range(config['experiment']['kernel_size_nums'])]
        self.rand_module = RandConvModule(
                          kernel_size=kernel_sizes,
                          mixing=config['experiment']['mix'],
                          identity_prob=config['experiment']['rand_conv_rate'],
                        #   rand_bias=args.rand_bias,
                        #   distribution=args.distribution,
                          )      
        
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        # if batch_idx == 1:
        #     print("date1: ", x[0,0,0])    
        #     print("date3: ", x[-1,0,0])    
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   


        self.rand_module.randomize()
        logits1 = self(self.rand_module(x))
        self.rand_module.randomize()
        logits2 = self(self.rand_module(x))
        
        p_clean, p_aug1, p_aug2 = F.softmax(logits, dim=1), F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        inv_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        return loss + self.consistency_loss_coeff*inv_loss
    
    
"""
convolution layer whose weights can be randomized

Created by zhenlinxu on 12/28/2019

https://github.com/wildphoton/RandConv/blob/main/lib/networks/rand_conv.py#L233
"""


class RandConvModule(nn.Module):
    def __init__(self, kernel_size=3, in_channels=2, out_channels=2,
                 rand_bias=False,
                 mixing=False,
                 identity_prob=0.0, distribution='kaiming_normal',
                 ):
        """

        :param net:
        :param kernel_size:
        :param in_channels:
        :param out_channels:
        :param rand_bias:
        :param mixing: "random": output = (1-alpha)*input + alpha* randconv(input) where alpha is a random number sampled
                            from a distribution defined by res_dist
        :param identity_prob:
        :param distribution:
        :param data_mean:
        :param data_std:
        :param clamp_output:
        """

        super(RandConvModule, self).__init__()
        # generate random conv layer
        print("Add RandConv layer with kernel size {}, output channel {}".format(kernel_size, out_channels))
        self.randconv = MultiScaleRandConv1d(in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_size,
                                             stride=1, rand_bias=rand_bias,
                                             distribution=distribution,
                                             )


        # mixing mode
        self.mixing = mixing # In the mixing mode, a mixing connection exists between input and output of random conv layer
        # self.res_dist = res_dist
        self.res_test_weight = None
        if self.mixing:
            assert in_channels == out_channels or out_channels == 1, \
                'In mixing mode, in/out channels have to be equal or out channels is 1'
            self.alpha = random.random()  # sample mixing weights from uniform distributin (0, 1)
        self.identity_prob = identity_prob  # the probability that use original input

    def forward(self, input):
        """assume that the input is whightened"""

        ######## random conv ##########
        if not (self.identity_prob > 0 and torch.rand(1) < self.identity_prob):
            # whiten input and go through randconv
            output = self.randconv(input)

            if self.mixing:
                output = (self.alpha*output + (1-self.alpha)*input)
        else:
            output = input

        return output

    def parameters(self, recurse=True):
        return self.randconv.parameters()

    def trainable_parameters(self, recurse=True):
        return self.randconv.trainable_parameters()

    # def whiten(self, input):
    #     return (input - self.data_mean) / self.data_std

    # def dewhiten(self, input):
    #     return input * self.data_std + self.data_mean

    def randomize(self):
        self.randconv.randomize()

        if self.mixing:
            self.alpha = random.random()

    def set_test_res_weight(self, w):
        self.res_test_weight = w

class RandConv1d(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, rand_bias=True,
                 distribution='kaiming_normal',
                #  clamp_output=None, range_up=None, range_low=None, 
                 **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param rand_bias:
        :param distribution:
        :param clamp_output:
        :param range_up:
        :param range_low:
        :param kwargs:
        """
        super(RandConv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=rand_bias, **kwargs)

        self.rand_bias = rand_bias
        self.distribution = distribution

        # self.clamp_output = clamp_output
        # self.register_buffer('range_up', None if not self.clamp_output else range_up)
        # self.register_buffer('range_low', None if not self.clamp_output else range_low)
        # if self.clamp_output:
        #     assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"


    def randomize(self):
        new_weight = torch.zeros_like(self.weight)
        with torch.no_grad():
            if self.distribution == 'kaiming_uniform':
                nn.init.kaiming_uniform_(new_weight, nonlinearity='conv1d')
            elif self.distribution == 'kaiming_normal':
                nn.init.kaiming_normal_(new_weight, nonlinearity='conv1d')
            elif self.distribution == 'kaiming_normal_clamp':
                fan = nn.init._calculate_correct_fan(new_weight, 'fan_in')
                gain = nn.init.calculate_gain('conv1d', 0)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    new_weight.normal_(0, std)
                    new_weight = new_weight.clamp(-2*std, 2*std)
            elif self.distribution == 'xavier_normal':
                nn.init.xavier_normal_(new_weight)
            else:
                raise NotImplementedError()

        self.weight = nn.Parameter(new_weight.detach())
        if self.bias is not None and self.rand_bias:
            # new_bias = self.bias.clone().detach()
            new_bias = torch.zeros_like(self.bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(new_bias, -bound, bound)
            self.bias = nn.Parameter(new_bias)

    def forward(self, input):
        output = super(RandConv1d, self).forward(input)

        # if self.clamp_output == 'clamp':
        #     output = torch.max(torch.min(output, self.range_up), self.range_low)
        # elif self.clamp_output == 'norm':
        #     output_low = torch.min(torch.min(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        #     output_up = torch.max(torch.max(output, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        #     output = (output - output_low)/(output_up-output_low)*(self.range_up-self.range_low) + self.range_low

        return output


class MultiScaleRandConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                 rand_bias=True, distribution='kaiming_normal',
                #  clamp_output=False, range_up=None, range_low=None, 
                **kwargs
                 ):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size: sequence of kernel size, e.g. (1,3,5)
        :param bias:
        """
        super(MultiScaleRandConv1d, self).__init__()

        # self.clamp_output = clamp_output
        # self.register_buffer('range_up', None if not self.clamp_output else range_up)
        # self.register_buffer('range_low', None if not self.clamp_output else range_low)
        # if self.clamp_output:
        #     assert (self.range_up is not None) and (self.range_low is not None), "No up/low range given for adjust"

        self.multiscale_rand_convs = nn.ModuleDict(
            {str(kernel_size): RandConv1d(in_channels, out_channels, kernel_size, padding = "same",
                                          rand_bias=rand_bias, distribution=distribution,
                                        #   clamp_output=self.clamp_output,
                                        #   range_low=self.range_low, range_up=self.range_up,
                                          **kwargs) for kernel_size in kernel_sizes})

        self.scales = kernel_sizes
        self.n_scales = len(kernel_sizes)
        self.randomize()

    def randomize(self):
        self.current_scale = str(self.scales[random.randint(0, self.n_scales-1)])
        self.multiscale_rand_convs[self.current_scale].randomize()

    def forward(self, input):
        output = self.multiscale_rand_convs[self.current_scale](input)
        return output