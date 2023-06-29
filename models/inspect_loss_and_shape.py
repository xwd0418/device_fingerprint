import torch

import numpy as np
from loss_module.MMD import mmd
if __name__ == "__main__":
    # 样本数量可以不同，特征数目必须相同

    # 100和90是样本数量，50是特征数目
    data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
    data_2 = torch.tensor(np.random.normal(loc=10,scale=10,size=(90,50)))
    print("MMD Loss:",mmd(data_1,data_2))

    data_1 = torch.tensor(np.random.normal(loc=0,scale=10,size=(100,50)))
    data_2 = torch.tensor(np.random.normal(loc=0,scale=9,size=(80,50)))

    print("MMD Loss:",mmd(data_1,data_2))
    
    d1 = torch.rand([16,512*8])
    d2 = torch.rand([16,512*8])
    print("MMD Loss:",mmd(d1,d2))


# MMD Loss: tensor(1.0866, dtype=torch.float64)
# MMD Loss: tensor(0.0852, dtype=torch.float64)