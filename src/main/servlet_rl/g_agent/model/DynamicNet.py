import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, in_dim, dropoutRate=0.1):
        super().__init__()

        self.l1 = nn.Linear(in_dim, in_dim * 2)
        self.l2 = nn.Linear(in_dim * 2, in_dim)
        self.dropoutRate = dropoutRate

    def forward(self, x):
        x0 = self.l1(x)
        x0 = F.relu(x0)
        x1 = x0 + F.dropout(x0, self.dropoutRate, training=self.training)
        x1 = self.l2(x1)
        x1 = F.dropout(x1, self.dropoutRate, training=self.training)
        return x1


class AttentionEncoder(nn.Module):
    def __init__(self, in_out_dim, num_heads=2, num_layers=1, dropoutRate=0.1):
        super().__init__()

        self.attn_block = nn.MultiheadAttention(in_out_dim, num_heads=num_heads,
                                                batch_first=True)  # _____
        self.ff = FeedForward(in_dim=in_out_dim, dropoutRate=dropoutRate)
        self.norm1 = nn.LayerNorm(in_out_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropoutRate = dropoutRate

    def forward(self, x, action_type_emb, actions, nodeTensor):
        macIds = nodeTensor[:, 2]  # __macId__
        jobIds = nodeTensor[:, 0]  # __jobId__
        opIds = nodeTensor[:, 1]  # __opId__
        attn_mask = create_attn_mask(macIds=macIds, jobIds=jobIds, opIds=opIds, actions=actions, batch_size=x.shape[0],
                                     seq_length=x.shape[1],
                                     head_num=self.num_heads,
                                     device=x.device.type)  # __macId___

        if self.training:
            mask = random_mask(batch_size=x.shape[0], seq_length=x.shape[1], device=x.device.type, mask_prob=0.1)
            # x_masked_fill_noise = random.randint(-10, 10) * 100000
            x_masked_fill_noise = -100.0
            x = x.masked_fill(torch.eq(mask, False).unsqueeze(2), float(x_masked_fill_noise))
            mask_action_type_emb = random_mask(batch_size=x.shape[0], seq_length=x.shape[1], device=x.device.type,
                                               mask_prob=0.1)
            # action_type_emb_masked_fill_noise = random.randint(-10, 10) * 100000
            action_type_emb_masked_fill_noise = -100.0
            action_type_emb = action_type_emb.masked_fill(torch.eq(mask_action_type_emb, False).unsqueeze(2),
                                                          float(action_type_emb_masked_fill_noise))
        for i in range(self.num_layers):
            x1, att_weight1 = self.attn_block(value=self.norm1(x), query=self.norm1(action_type_emb + x),
                                              key=self.norm1(action_type_emb + x),
                                              attn_mask=attn_mask)
            x2 = x + F.dropout(x1, self.dropoutRate, training=self.training)
            x3 = self.ff(x2)
            x = x2 + x3
        return x


class DynamicNet(torch.nn.Module):

    def __init__(self, in_out_dim, nLayer=1, nhead=2):
        super().__init__()
        # _______，_____________
        action_type_size = 2
        self.action_type_encoder = nn.Embedding(action_type_size, in_out_dim)  # ______________
        self.attn_encoder1 = AttentionEncoder(in_out_dim, num_heads=nhead,
                                              num_layers=nLayer)
        self.attn_encoder2 = AttentionEncoder(in_out_dim, num_heads=nhead,
                                              num_layers=nLayer)
        self.out_dim = in_out_dim

    def forward(self, nodeTensor: torch.Tensor, x: torch.Tensor, action: torch.Tensor, need_rand_mask=True):
        # ________
        action_type_emb = self.action_type_encoder(action)
        # x, action_type_emb, nodeTensor
        x = self.attn_encoder1(x=x, action_type_emb=action_type_emb, actions=action, nodeTensor=nodeTensor)
        x = self.attn_encoder2(x=x, action_type_emb=action_type_emb, actions=action, nodeTensor=nodeTensor)

        return x


def create_attn_mask(macIds, jobIds, opIds, actions, batch_size, seq_length, head_num, device="cpu"):
    # 1.
    macId_indices = torch.ones(seq_length, seq_length, dtype=torch.bool).to(device)
    for nodeId in range(seq_length):
        macId_indices[nodeId] = ~ (macIds[nodeId] == macIds)
    # 2.
    jobId_indices = torch.ones(seq_length, seq_length, dtype=torch.bool).to(device)
    for nodeId in range(seq_length):
        # _______jobId______opMax_!_______!
        jobId_indices[nodeId] = ~ (jobIds[nodeId] == jobIds)

    one_attn_mask = torch.logical_and(macId_indices, jobId_indices)
    # 3.
    # attn_mask = torch.ones(batch_size * head_num, seq_length, seq_length, dtype=torch.bool)
    attn_mask = ~actions.to(torch.bool).unsqueeze(1).unsqueeze(-1).repeat(1, head_num, 1, seq_length).view(
        batch_size * head_num,
        seq_length, seq_length)

    for batchIdx in range(batch_size):
        # one_attn_action_mask = one_attn_mask.clone()
        # # # # todo ______
        # for actionIdx in range(seq_length):
        #     if actions[batchIdx][actionIdx] == 1:
        #         one_attn_action_mask[actionIdx] = torch.zeros(seq_length, dtype=torch.bool).to(device)
        #         one_attn_action_mask_T = one_attn_action_mask.transpose(0, 1)
        #         one_attn_action_mask_T[actionIdx] = torch.zeros(seq_length, dtype=torch.bool).to(device)
        #         one_attn_action_mask = one_attn_action_mask_T.transpose(0, 1)
        for head_numIdx in range(head_num):
            # attn_mask[batchIdx * head_num + head_numIdx] = one_attn_action_mask
            # todo _____,______
            attn_mask[batchIdx * head_num + head_numIdx] = torch.logical_and(one_attn_mask, attn_mask[
                batchIdx * head_num + head_numIdx])

    attn_mask = attn_mask.to(device)
    return attn_mask


def random_mask(batch_size, seq_length, mask_prob=0.1, device="cpu"):
    """
    ___________，__Transformer_____。

    __:
    - batch_size: ____。
    - seq_length: ____。
    - mask_prob: _____，________0（__）___，___0.2。

    __:
    - ____，___(batch_size, seq_length)，__1_____，0____。
    """
    mask = torch.bernoulli(torch.full(size=(batch_size, seq_length), fill_value=1 - mask_prob)).to(torch.bool).to(
        device)
    return mask
