import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.main2.servlet.ConfigManager import ConfigManager
from torch import Tensor

from src.main.servlet_rl.g_agent.model.CommonNet import FCN
from src.main.servlet_rl.g_agent.model.StructureEncoder import StructureEncoder
from src.main.utils.Logger import logprint


def get_positions_encoding(positions: Tensor, d_model: int):
    """
    ___________
    :param positions: _________1_PyTorch__
    :param d_model: ____
    :return: ___________，___(positions.size(0), d_model)
    """
    assert isinstance(positions, torch.Tensor) and positions.dim() == 1, "positions___1___"
    assert isinstance(d_model, int) and d_model > 0, "d_model______"
    # _________
    pe = torch.zeros(positions.size(0), d_model).to(positions.device)

    # _____
    position_encodings = torch.arange(0, d_model, step=2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    position_encodings = torch.outer(positions.float(), position_encodings.to(positions.device))  # ________

    # _________
    pe[:, 0::2] = torch.sin(position_encodings)
    pe[:, 1::2] = torch.cos(position_encodings)

    return pe


class NodeEncoder(torch.nn.Module):
    def __init__(self, attr_dim: int = 10, dropoutRate=0.5, attrN: int = 5):
        """
        Args:
            attr_dim: _____，______
            dropoutRate: Dropout__，___0_1__
        """
        super().__init__()

        # ____

        assert isinstance(attr_dim, int) and attr_dim > 0, "attr_dim______"
        assert 0 <= dropoutRate <= 1, "dropoutRate___0_1__"
        self.attrN = attrN
        self.attr_dim = attr_dim
        self.dropoutRate = dropoutRate
        # self.x_output_dim = attr_dim * attrN

        self.attr_type_encoder = nn.Embedding(attrN, embedding_dim=attr_dim)

        # ______tensor,0:jobId,1:opId,2:macId,3:execId, 4:time

        self.digits = nn.Parameter(torch.tensor([digit for digit in range(attrN)], dtype=torch.int),
                                   requires_grad=False)

    def forward(self, x):
        """
        "node.jobId, node.opId, node.macId, node.execId, node.time,+______"

        ______，___x____
        # __：________________（_____________/_______，_____）
        # example_node_encoder = NodeEncoder(attr_dim=10, dropoutRate=0.5)
        # fake_input = torch.randn(10, 4)  # _______10___，_____4___
        # output = example_node_encoder(fake_input)
        # output.shape  # ____: torch.Size([10, 40])
        """
        try:
            assert x.shape[1] >= self.attrN, f"__x______{self.attrN}"
        except:
            logprint.warning(x.shape)
        # ___________
        encoded_digits = []
        for i in range(self.attrN):
            digit_encoding = get_positions_encoding(x[:, i], self.attr_dim)
            attr_typeTsr = self.attr_type_encoder(self.digits[i])

            digit_encoding += attr_typeTsr
            encoded_digits.append(digit_encoding)
        #####################################################################################################
        # _________

        # ______
        available_mac_encodings = get_positions_encoding(torch.arange(0, x.shape[1] - self.attrN),
                                                         d_model=self.attr_dim).unsqueeze(1).repeat(1, x.shape[0], 1).to(x.device)
        available_mac_typeTsr = self.attr_type_encoder(self.digits[2]).unsqueeze(0).unsqueeze(0).repeat(
            x.shape[1] - self.attrN, x.shape[0], 1)
        available_mac_encodings = available_mac_encodings + available_mac_typeTsr
        # _________
        available_time_encodings = get_positions_encoding(x[:, self.attrN:].transpose(0, 1).reshape(-1),
                                                          self.attr_dim).reshape(
            x.shape[1] - self.attrN,
            x.shape[0], self.attr_dim).to(x.device)
        available_time_typeTsr = self.attr_type_encoder(self.digits[4]).unsqueeze(0).unsqueeze(0).repeat(
            x.shape[1] - self.attrN, x.shape[0], 1)
        available_time_encodings = available_time_typeTsr + available_time_encodings
        # todo ____-start
        # available_mac_encodings=torch.zeros_like(available_mac_encodings)
        # available_time_encodings=torch.zeros_like(available_time_encodings)
        # todo ____-end


        return encoded_digits, (available_mac_encodings, available_time_encodings)

    def miniBatch(self):
        "_______,_________"
        pass


class EdgeEncoder(torch.nn.Module):
    def __init__(self, edge_attr_dim=10, dropoutRate=0.1):
        """

        Args:
            attr_dim: _____
            deviceName: ___
        """
        super().__init__()
        # self.deviceName = deviceName
        self.edge_attr_dim = edge_attr_dim
        self.dropoutRate = dropoutRate
        self.attr_type_encoder = nn.Embedding(2, embedding_dim=edge_attr_dim)
        self.edge_attr_output_dim = edge_attr_dim

    def forward(self, edge_attr: Tensor):
        """

        Args:
            edge_attr: __[[],[]],_____

        Returns:

        """
        "edge_attr:[edgeType:{0-job,1-mac},jobId or macId]"
        # assert edge_attr.shape[1] >= 2, "__edge_attr______2"
        x0 = self.attr_type_encoder(edge_attr[:, 0])
        x1 = get_positions_encoding(edge_attr[:, 1], self.edge_attr_dim)
        x = x0 + x1
        # x = F.dropout(x, p=self.dropoutRate, training=self.training)

        return x


class SeqEncoder(nn.Module):
    def __init__(self, y_dim, x_dim, h_dim, num_layers, bidirection, batch, deviceName):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.bidirection = bidirection
        self.batch = batch
        self.lstm = nn.LSTM(input_size=self.x_dim, hidden_size=self.h_dim,
                            num_layers=self.num_layers,
                            batch_first=True, bidirectional=False)
        self.h0 = torch.zeros(self.num_layers * self.bidirection, self.batch, self.h_dim).to(deviceName)
        self.c0 = torch.zeros(self.num_layers * self.bidirection, self.batch, self.h_dim).to(deviceName)
        # _____
        seqLayerDim = self.num_layers * self.bidirection * self.h_dim
        self.adjustShape = FCN(x_dim=seqLayerDim, y_dim=y_dim, nLayer=2)
        # self.seqLayers.append(adjustShape)

    def forward(self, actionList: Tensor):
        # state:batch,x_size,x_dim
        # x.shape:(batch,x_size,x_dim)
        # h0.shape:(num_layers * bidirection,batch,h_dim)
        # c0.shape:(num_layers * bidirection,batch,h_dim)
        # ht.shape:(batch, x_size, h_dim) __h_____
        # hn.shape:(num_layers*bidirection, batch, h_dim])__h_____
        # cn.shape:(num_layers*bidirection, batch, h_dim)__c_____
        ht, (hn, cn) = self.lstm(actionList, (self.h0, self.c0))
        hn1D = hn.view(1, -1)

        y = self.adjustShape(hn1D)
        return y.view(-1)


class MacSeqEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers=1):
        super().__init__()
        trm = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=1, batch_first=True)
        norm = nn.LayerNorm(hidden_channels)

        self.seq_encoder = nn.TransformerEncoder(trm, num_layers=num_layers, norm=norm)
        self.out_dim = hidden_channels

    def forward(self, x: Tensor):
        x = F.layer_norm(x, x.size())
        x = F.dropout(F.relu(x), p=0.4,
                      training=self.training)
        x = self.seq_encoder(x)
        return x


class JobSeqEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers=1):
        super().__init__()
        trm = nn.TransformerEncoderLayer(d_model=hidden_channels, activation=F.relu, nhead=1, batch_first=True)
        norm = nn.LayerNorm(hidden_channels)
        self.seq_encoder = nn.TransformerEncoder(trm, num_layers=num_layers, norm=norm)
        self.out_dim = hidden_channels

    def forward(self, x: Tensor):
        x = F.layer_norm(x, x.size())
        x = F.dropout(F.relu(x), p=0.4,
                      training=self.training)
        x = self.seq_encoder(x)
        return x


class RepresentNet(torch.nn.Module):

    def __init__(self,
                 attr_dim=10,
                 attrN=5,
                 edge_attr_dim=10):
        super().__init__()
        self.nodeEncoder = NodeEncoder(attr_dim=attr_dim, attrN=attrN)
        self.edgeEncoder = EdgeEncoder(edge_attr_dim=edge_attr_dim)
        self.structureEncoder = StructureEncoder(hidden_channels=attr_dim * 5,
                                                 edge_attr_dim=self.edgeEncoder.edge_attr_dim)
        # self.jobSeqEncoder = JobSeqEncoder(hidden_channels=attr_dim * 3)
        # self.macSeqEncoder = MacSeqEncoder(hidden_channels=attr_dim * 3)
        # self.unet= GraphUNet(in_channels=out_channels, out_channels=out_channels,depth=10,pool_ratios=0.5,sum_res=True,act='relu')
        self.no_edge_typeTsr = nn.Parameter(torch.tensor([[], []], dtype=torch.int), requires_grad=False)
        self.no_edge_attrTsr = nn.Parameter(torch.tensor([[]], dtype=torch.float32), requires_grad=False)
        self.attr_dim = attr_dim
        self.out_dim = attr_dim * 5

    def forward(self, x: List[Tensor], edge_index: Tensor, edge_attr: Tensor, actionList: Tensor = None):
        # 1.node___
        encoded_digits, available_mac_time_encodings = self.nodeEncoder(x)
        # 2.structureEncoder
        x = torch.cat([encoded_digits[0], encoded_digits[1], encoded_digits[2], encoded_digits[3], encoded_digits[4]],
                      dim=1)
        # x = torch.cat([encoded_digits[0], encoded_digits[1], encoded_digits[4]], dim=1)
        # if edge_attr.shape[0] > 0 else self.no_edge_attrTsr
        edge_attr = self.edgeEncoder(edge_attr)
        # ________
        x = self.structureEncoder(x, edge_index, edge_attr=edge_attr)

        # 3.__
        # mask_indices = None
        # if self.training:
        #     mask_ratio = 0.15  # __15%___
        #     mask_indices = torch.rand(size=(1, x.shape[0], 1), device="cuda") < mask_ratio
        # 4.jobSeqEncoder
        # x_jobSeq = torch.cat([encoded_digits[0], encoded_digits[1], encoded_digits[4]], dim=1)
        # x_jobSeq = x_jobSeq.unsqueeze(0)
        # # ________，________float("0")
        # if self.training:
        #     x_jobSeq = x_jobSeq.masked_fill(mask=mask_indices == True,
        #                                     value=float("-500"))
        # x1 = self.macSeqEncoder(x + x_jobSeq)
        # x = x1.squeeze(0)
        # 5.macSeqEncoder
        # x_macSeq = torch.cat([encoded_digits[2], encoded_digits[3], encoded_digits[4]], dim=1)
        # x_macSeq = x_macSeq.unsqueeze(0)
        # if self.training:
        #     x_macSeq = x_macSeq.masked_fill(mask=mask_indices == True,
        #                                     value=float("-500"))
        # x1 = self.macSeqEncoder(x + x_macSeq)
        # x = x1.squeeze(0)
        #
        # x = F.relu(x)
        # x = F.layer_norm(x, x.size())
        # x = F.dropout(x, p=0.4, training=self.training)
        return x, available_mac_time_encodings
