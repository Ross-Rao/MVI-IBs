"""
@article{zheng2020end,
  title={End-to-end object detection with adaptive clustering transformer},
  author={Zheng, Minghang and Gao, Peng and Wang, Xiaogang and Li, Hongsheng and Dong, Hao},
  journal={arXiv preprint arXiv:2011.09315},
  year={2020}
}
"""

# python import
import torch
import torch.nn as nn
# local import
# if you use cuda extension, please ensure data in cuda device.
from extensions import broadcast, weighted_sum


# from extensions import torch_broadcast as broadcast
# from extensions import torch_weighted_sum as weighted_sum


class WeightedSoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _weighted_softmax(x, dim=None, weight=None):
        ret = torch.softmax(x, dim=dim)
        if weight is not None:
            ret = ret * weight.unsqueeze(1)
            ret = ret / ret.sum(dim=-1, keepdim=True)
        return ret

    def forward(self, x, dim=None, weight=None):
        return self._weighted_softmax(x, dim=dim, weight=weight)


class CalcCenter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clusters, counts):
        # original implementation might be wrong? if counts=0, then weights=inf
        weights = 1 / counts.float()  # cluster-level weights
        # weights = counts.float() / counts.sum(dim=-1, keepdim=True).float()  # sample-level weights
        ctx.save_for_backward(clusters.clone(), weights.clone())
        center = weighted_sum(x, clusters, weights)

        return center

    @staticmethod
    def backward(ctx, *grad_output):
        clusters, weights = ctx.saved_tensors
        grad_center = grad_output[0]
        grad_input = broadcast(grad_center, clusters, weights)
        return grad_input, None, None


class Broadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center, clusters):
        b, c, d = center.shape
        ctx.save_for_backward(clusters.clone(), torch.tensor(c))
        weights = torch.ones((b, c), dtype=torch.float, device=center.device)
        x = broadcast(center, clusters, weights)
        return x

    @staticmethod
    def backward(ctx, *grad_output):
        grad = grad_output[0]
        b, n, d = grad.shape
        clusters = ctx.saved_tensors[0]
        c = ctx.saved_tensors[1].int()
        weights = torch.ones((b, c), dtype=torch.float, device=grad.device)
        grad_center = weighted_sum(grad, clusters, weights)
        return grad_center, None, None


class AdaClusteringAttention(nn.Module):
    """Use E2LSH to adaptively cluster queries or keys

    Arguments
    ---------
        group_q: If true, use E2LSH to adaptively cluster queries
        group_k: If true, use E2LSH to adaptively cluster keys
        softmax_temp: The temperature to use for the softmax attention
        attention_dropout: The dropout rate to apply to the attention
    """

    def __init__(self, group_q=False, group_k=False, softmax_temp=1, number_clusters=128, attention_dropout=0.0):
        super(AdaClusteringAttention, self).__init__()
        self.group_q = group_q
        self.group_k = group_k
        self.softmax_temp = softmax_temp
        self.number_clusters = number_clusters
        if attention_dropout > 0.0:
            self.dropout = nn.Dropout(attention_dropout)
        else:
            self.dropout = None
        self.softmax = WeightedSoftMax()

        self.q_clusters = None
        self.k_clusters = None

    def _create_clusters(self, x, clusters):
        """
        count the number of clusters as weights
        """
        b, n, d = x.shape

        groups = clusters.int()
        c = self.number_clusters + 1
        counts = torch.zeros((clusters.shape[0], c), dtype=torch.int, device=clusters.device)
        for i in range(0, groups.shape[0]):
            counts[i, :] = groups[i, :].bincount(minlength=c)
        groups = groups.repeat(int(b / clusters.shape[0]), 1)
        counts = counts.repeat(int(b / clusters.shape[0]), 1)

        return groups, counts.contiguous()

    def forward(self, queries, keys, values, clusters, key_padding_mask=None):

        if self.group_q:
            q_groups, q_counts = self._create_clusters(queries, clusters)
            q_center = CalcCenter.apply(queries, q_groups, q_counts)
            self.q_clusters = q_counts.size(-1)  # number of clusters
        else:
            q_groups = None
            q_center = queries

        if self.group_k:
            k_groups, k_counts = self._create_clusters(keys, clusters)
            k_center = CalcCenter.apply(keys, k_groups, k_counts)
            v_center = CalcCenter.apply(values, k_groups, k_counts)
            self.k_clusters = k_counts.size(-1)  # number of clusters
        else:
            k_counts = None
            k_center = keys
            v_center = values

        qk = torch.bmm(q_center, k_center.permute(0, 2, 1))
        if key_padding_mask is not None:
            assert self.group_k is not True
            qk = qk.view(key_padding_mask.size(0), -1, q_center.size(1), keys.size(1))
            qk = qk.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            qk = qk.view(-1, q_center.size(1), keys.size(1))

        a_full = self.softmax(self.softmax_temp * qk, dim=-1, weight=k_counts)
        if self.dropout:
            a = self.dropout(a_full)
        else:
            a = a_full

        v = torch.bmm(a, v_center)
        if self.group_q:
            v = Broadcast.apply(v, q_groups)

        return v.contiguous(), a_full[:, :, 0]


if __name__ == '__main__':
    import torch
    from torch import nn

    B, N, D = 32, 32, 512
    q = torch.randn(B, N, D).cuda()
    clus = torch.randint(0, 128, (B, N)).cuda()

    attention = AdaClusteringAttention(group_q=False, group_k=True, softmax_temp=1,
                                       number_clusters=128, attention_dropout=0.0)
    out, attn = attention(q, q, q, clus)
    print(out.shape, attn.shape)
    print(out)
    print(attn)

