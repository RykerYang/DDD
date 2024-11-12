import torch
import torch.nn as nn
from torch import Tensor
from src.main.servlet_rl.g_agent.model.Unet import UNet
import torch.nn.functional as F


# class GraphUNetLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GraphUNetLayer, self).__init__()
#         self.conv1 = GCNConv(in_channels, out_channels)
#         self.conv2 = GCNConv(out_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         """
#         Args:
#             x: ____ (num_nodes, in_channels)
#             edge_index: _____ (2, num_edges)
#
#         Returns:
#             ____ (num_nodes, out_channels)
#         """
#         x = torch.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
#
#
# class UNet(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(UNet, self).__init__()
#         self.encoder = nn.ModuleList([
#             GraphUNetLayer(in_channels if i == 0 else hidden_channels, hidden_channels)
#             for i in range(4)  # __4_____
#         ])
#         self.decoder = nn.ModuleList([
#             GraphUNetLayer(hidden_channels * 2 if i == 0 else hidden_channels, hidden_channels)
#             for i in range(3)  # __3_____，____________
#         ])
#         self.final_conv = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index):
#         skips = []
#         for layer in self.encoder:
#             x = layer(x, edge_index)
#             skips.append(x)
#             x = torch.unsqueeze(x, dim=0)  # _________，_________，___
#
#         # _____，______，____________________
#         for layer in reversed(self.decoder):
#             skip_feature = skips.pop()
#             x = torch.cat((x, skip_feature), dim=1)  # ____
#             x = layer(x, edge_index)
#
#         x = self.final_conv(x, edge_index)
#         return x


class StructureEncoder(nn.Module):
    def __init__(self, hidden_channels=64, edge_attr_dim=6, num_layers=2, num_graphUNetList=1, pool_ratios=0.1):
        # todo def __init__(self, hidden_channels=64, edge_attr_dim=6, num_layers=1, num_graphUNetList=1, pool_ratios=0.3):
        """
        __RGCNConv________, _____.
        Args:
            hidden_channels(int): ___(__,__,___)_______
            num_layers: _____
            edgeTypeN: ______,_________________
        """
        super().__init__()
        # GraphUNet
        self.num_graphUNetList = num_graphUNetList
        # 1.
        self.graphUNetList = nn.ModuleList([UNet(in_channels=hidden_channels, hidden_channels=hidden_channels,
                                                 out_channels=hidden_channels, depth=num_layers,
                                                 pool_ratios=pool_ratios, sum_res=True,
                                                 edge_attr_dim=edge_attr_dim) for i in range(num_graphUNetList)])
        # 2.
        # unet = UNet(in_channels=hidden_channels, hidden_channels=hidden_channels,
        #             out_channels=hidden_channels, depth=num_layers,
        #             pool_ratios=pool_ratios, sum_res=True,
        #             edge_attr_dim=edge_attr_dim)
        # self.graphUNetList = nn.ModuleList([unet for i in range(num_graphUNetList)])

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        # __________:layer(x, torch.tensor([[],[]],dtype=torch.int), torch.tensor([]))
        # x = F.dropout(x, p=0.4,
        #               training=self.training)

        # todo _____
        edge_attr = torch.zeros_like(edge_attr)
        # __
        for i in range(self.num_graphUNetList):
            noisy = self.graphUNetList[i](x, edge_index=edge_index, edge_attr=edge_attr)
            x = x + F.dropout(noisy, p=0.5, training=self.training)
        # x = self.graphUNet(x, edge_index=edge_index)
        # x=self.gat(x, edge_index=edge_index, edge_attr=edge_attr)

        return x

#######################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
# from torch_geometric.nn import DeepGCNLayer, GATv2Conv
#
#
# class StructureEncoder(nn.Module):
#     def __init__(self, hidden_channels=64, num_layers=6, edge_attr_dim=6):
#         """
#         __RGCNConv________, _____.
#         Args:
#             hidden_channels(int): ___(__,__,___)_______
#             num_layers: _____
#             edgeTypeN: ______,_________________
#         """
#         super().__init__()
#         # trm = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=4, batch_first=True)
#         # self.trm_encoder = nn.TransformerEncoder(trm, num_layers=1)
#
#         self.convList = nn.ModuleList()
#         # GCN_
#         conv0 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, heads=10,
#                          edge_dim=edge_attr_dim, aggr='softmax', concat=False)
#         conv1 = GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, heads=10,
#                          edge_dim=edge_attr_dim, aggr='softmax', concat=False)
#         for i in range(1, num_layers + 1):
#
#
#             norm = nn.LayerNorm(hidden_channels)
#             act = nn.ReLU()
#             if i%2==0:
#                 conv = conv0
#             else:
#                 conv = conv1
#             layer = DeepGCNLayer(conv, act=act, block='res+', dropout=0.4, norm=norm)
#             self.convList.append(layer)
#
#     def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
#         # __________:layer(x, torch.tensor([[],[]],dtype=torch.int), torch.tensor([]))
#
#         x = F.dropout(x, p=0.4,
#                       training=self.training)
#         for idx, layer in enumerate(self.convList[0:]):
#             # x = F.layer_norm(x, x.size())
#             x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
#
#         #
#         # mask_indices = None
#         # if self.training:
#         #     mask_ratio = 0.15  # __15%___
#         #     mask_indices = torch.rand(size=(1, x.shape[0], 1), device="cuda") < mask_ratio
#         # x = x.unsqueeze(0)
#         # if self.training:
#         #     x = x.masked_fill(mask=mask_indices == True,
#         #                       value=float("-500"))
#         # x = self.trm_encoder(x)
#         # x = x.squeeze(0)
#
#         return x
