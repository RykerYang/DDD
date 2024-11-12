from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn
from src.main.servlet_rl.g_agent.model.CommonNet import FCN
from src.main.servlet_rl.i_rlto.Action import Action


class Head(torch.nn.Module):
    def __init__(self, in_dim, out_dim, nLayer):
        super().__init__()
        # self.config = config
        # in_dim = config.in_dim
        # out_dim = config.out_dim
        # # default:5
        # nLayer = config.nLayer
        self.fc = FCN(x_dim=in_dim, y_dim=out_dim, lastLayerNeedActive=0, nLayer=nLayer)
        # self.deviceName: str = deviceName

    def forward(self, x):
        y = self.fc(x)

        return y


# class PredictNetConfig(Config):
#     def __init__(self):
#         self.hidden_channels = 64
#         # class Head
#         self.in_dim: int
#         self.out_dim: int
#         # default:5
#         self.nLayer = 5
#         # /class Head


class PredictNet(torch.nn.Module):

    def __init__(self,
                 # phead_qhead_____
                 deviceName: str,
                 pheadConfig: Dict,
                 qheadConfig: Dict, attr_dim=10):
        super().__init__()
        self.attr_dim = attr_dim
        self.phead = Head(**pheadConfig)
        self.qhead = Head(**qheadConfig)
        self.available_mac_time_encoder = nn.Linear(2 * attr_dim, pheadConfig["in_dim"])
        self.action_merge_encoder = nn.ModuleList()
        for i in range(2):
            self.action_merge_encoder.append(nn.Linear(pheadConfig["in_dim"], pheadConfig["in_dim"]))

        action_type_size = 2
        self.action_type_encoder = nn.Embedding(action_type_size, pheadConfig["in_dim"])  # ______________
        self.x_weight = nn.Parameter(torch.ones(1))
        self.avaliable_mac_weight = nn.Parameter(torch.ones(1))
        # self.avaliable_time_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, actionTable: List[Action], available_mac_time_encodings):
        # 1._x________
        available_mac_encodings = available_mac_time_encodings[0]
        available_time_encodings = available_mac_time_encodings[1]
        # available_mac_encodings = available_mac_time_encodings[0]
        # available_time_encodings =available_mac_time_encodings[1]

        # available_mac_time_tensors = available_mac_encodings + available_time_encodings
        # available_mac_time_tensors = available_mac_encodings
        # available_mac_time_tensors = self.available_mac_time_encoder(available_mac_time_tensors)
        x = torch.cat([x * self.x_weight, self.avaliable_mac_weight * torch.mean(available_mac_encodings, dim=0)], dim=1)
        # 2.______
        # #########################################################################
        # todo ____action________
        # -_
        index = torch.tensor([[action.nodeIdxA, action.nodeIdxB] for action in actionTable]).to(x.device).view(-1)
        # ________
        index = index.unsqueeze(1)
        # ____
        index = index.expand(index.size(0), x.size(1))
        y = x.gather(0, index).unsqueeze(1).view(-1, 2, x.size(1))
        ##########################
        # -_
        # need_action_info = True
        # if need_action_info == False:
        #     y = x.unsqueeze(0)
        #     y = y.expand(len(actionTable), y.size(1), y.size(2))
        ##########################################################################

        y = torch.mean(F.relu(y), dim=1)
        # 3.pq__
        p = self.phead(y)
        p = F.softmax(p, dim=0).squeeze(1)
        # logprint.info("torch.sum(p):{}".format(torch.sum(p)))
        q = self.qhead(y).squeeze(1)

        return p, q
