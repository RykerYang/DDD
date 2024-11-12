from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class FCN(torch.nn.Module):
    # n_FC,x_dim=128,y_dim=1,nLayer=5,times=3,  x_dim=192, y_dim=1,
    def __init__(self, x_dim, times=3, y_dim=64, dropoutRate=0.5, nLayer=1, lastLayerNeedActive=1):
        super().__init__()
        if nLayer < 0:
            raise ValueError()
        # 1.__________
        parameterList: List[int] = [x_dim, y_dim]
        for i in range(0, nLayer - 1):
            parameterList.insert(i * 2 + 1, y_dim * times)
            parameterList.insert(i * 2 + 1, y_dim * times)
        # 2._______
        self.node_encoder = torch.nn.ModuleList()
        for i in range(0, nLayer):
            fc = nn.Sequential(nn.Linear(parameterList[i * 2], parameterList[i * 2 + 1]),
                               nn.ReLU(),
                               nn.Dropout(p=dropoutRate)
                               )
            if i == nLayer - 1:
                if lastLayerNeedActive == 0:
                    fc = nn.Sequential(nn.Linear(parameterList[i * 2], parameterList[i * 2 + 1]),
                                       nn.Dropout(p=dropoutRate)
                                       )

            self.node_encoder.append(fc)

    def forward(self, x: Tensor):
        for module in self.node_encoder:
            x1 = x
            x2 = module(x)
            # ____
            if x1.shape == x2.shape:
                x = x1 + x2
            else:
                x = x2

        return x
