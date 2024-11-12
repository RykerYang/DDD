import torch.nn.functional as F
from torch import nn


class FitnessNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.l1 = nn.Linear(in_dim, in_dim // 2 + 1)
        self.l2 = nn.Linear(in_dim // 2 + 1, 1)
        # self.l3 = nn.Linear(dim, 1)

    def forward(self, x):
        x0 = self.l1(x)
        x0 = F.relu(x0)
        # logprint.info(f"FNN_x0,x.mean:{x0.mean()},x.std:{x0.std()}")
        # boardprint.scalar("eval/output_analysis", {f"FNN_x0": x0.mean().item()})
        x0 = x0 + F.dropout(F.layer_norm(x0, x0.shape), 0.5, training=self.training)

        x1 = self.l2(x0)
        # x1 = F.softmax(x1, dim=2)
        return x1
