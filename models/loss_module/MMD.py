
import torch
import torch.nn as nn
import numpy as np

# def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     """计算Gram核矩阵
#     source: sample_size_1 * feature_size 的数据
#     target: sample_size_2 * feature_size 的数据
#     kernel_mul: 这个概念不太清楚 感觉也是为了计算每个核的bandwith
#     kernel_num: 表示的是多核的数量
#     fix_sigma: 表示是否使用固定的标准差
#         return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
#                         矩阵，表达形式:
#                         [   K_ss K_st
#                             K_ts K_tt ]
#     """
#     n_samples = int(source.size()[0])+int(target.size()[0])
#     total = torch.cat([source, target], dim=0) # 合并在一起

#     total0 = total.unsqueeze(0).expand(int(total.size(0)), \
#                                        int(total.size(0)), \
#                                        int(total.size(1)))
#     total1 = total.unsqueeze(1).expand(int(total.size(0)), \
#                                        int(total.size(0)), \
#                                        int(total.size(1)))
#     L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

#     # 计算多核中每个核的bandwidth
#     if fix_sigma:
#         bandwidth = fix_sigma
#     else:
#         bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#     bandwidth /= kernel_mul ** (kernel_num // 2)
#     bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

#     # 高斯核的公式，exp(-|x-y|/bandwith)
#     kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
#                   bandwidth_temp in bandwidth_list]

#     return sum(kernel_val) # 将多个核合并在一起

# def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
#     # print("source shape: \n\n\n\n\n", source.shape)
#     n = int(source.size()[0])
#     m = int(target.size()[0])

#     kernels = guassian_kernel(source, target,
#                               kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
#     XX = kernels[:n, :n] 
#     YY = kernels[n:, n:]
#     XY = kernels[:n, n:]
#     YX = kernels[n:, :n]

#     XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
#     XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

#     YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
#     YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
    	
#     loss = (XX + XY).sum() + (YX + YY).sum()
#     return loss


class MMD_loss(nn.Module):
    def __init__(self, MMD_sample_size=float('inf'), kernel_mul = 2.0, kernel_num = 5, fix_sigma=None,):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.MMD_sample_size = MMD_sample_size
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def calc_mmd(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss 
    
    def forward(self, source, target):
        batch_size = source.shape[0]//2
        if self.MMD_sample_size>=batch_size:
            return self.calc_mmd(source,target)
        sample_num = batch_size//self.MMD_sample_size
        total_mmd = 0
        reversed_target = torch.flip(target, [0])
        for i in range(sample_num):
            total_mmd += self.calc_mmd(source[i*self.MMD_sample_size:(i+1)*self.MMD_sample_size],
                                   reversed_target[i*self.MMD_sample_size:(i+1)*self.MMD_sample_size])
        return total_mmd/sample_num
        
