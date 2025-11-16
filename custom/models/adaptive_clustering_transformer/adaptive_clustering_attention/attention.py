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
if __package__ in [None, ""]:
    from extensions import broadcast, weighted_sum
    # for testing, you can use the following code
    # from extensions import torch_broadcast as broadcast
    # from extensions import torch_weighted_sum as weighted_sum
else:
    from .extensions import broadcast, weighted_sum  # used in outer import


__all__ = ['AdaptiveClusteringAttention']


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
    def inf(x, clusters, counts):
        # original implementation might be wrong? if counts=0, then weights=inf
        # weights = 1 / counts.float()  # cluster-level weights
        weights = torch.where(counts != 0, 1 / counts.float(), torch.zeros_like(counts.float()))
        # weights = counts.float() / counts.sum(dim=-1, keepdim=True).float()  # sample-level weights
        return weighted_sum(x, clusters, weights)

    @staticmethod
    def forward(ctx, x, clusters, counts):
        weights = torch.where(counts != 0, 1 / counts.float(), torch.zeros_like(counts.float()))
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
    def inf(center, clusters):
        b, c, d = center.shape
        weights = torch.ones((b, c), dtype=torch.float, device=center.device)
        x = broadcast(center, clusters, weights)
        return x

    @staticmethod
    def forward(ctx, center, clusters):
        b, c, d = center.shape
        ctx.save_for_backward(clusters.clone(), torch.tensor(c))
        x = Broadcast.inf(center, clusters)
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


class AdaptiveClusteringAttention(nn.Module):
    """
    Example:
        >>> import torch
        >>> from torch import nn

        >>> B, N, D = 32, 32, 512
        >>> q = torch.randn(B, N, D).cuda()
        >>> clus = torch.randint(0, 128, (B, N)).cuda()
        >>> multi_attention = AdaptiveClusteringAttention(
        ...     dim=D, num_heads=8, attention_dropout=0.1, projection_dropout=0.1,
        ...     group_q=False, group_k=True, softmax_temp=1, number_clusters=128
        ... ).to(q.device)
        >>> out = multi_attention(clus, q)
        >>> print(out.shape)
        torch.Size([32, 32, 512])
        >>> # print(out)
    """

    def __init__(self,
                 dim,
                 num_heads,
                 attention_dropout=0.1,
                 projection_dropout=0.1,
                 group_q=False,
                 group_k=True,
                 softmax_temp=1,
                 number_clusters=128):
        super().__init__()
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)

        # Adaptive Clustering
        self.group_q = group_q
        self.group_k = group_k
        self.softmax_temp = softmax_temp
        self.softmax = WeightedSoftMax()
        self.number_clusters = number_clusters

        self.autograd = nn.Parameter(torch.tensor(1.0))  # used to control autograd

    def forward(self, cluster, q, kv=None):
        if kv is None:
            kv = q
        (b, n, c), h = q.shape, self.num_heads

        # (b, n, c) -> (b, h, n, e)
        q = self.w_q(q).reshape(b, n, h, -1).transpose(1, 2)
        kv = self.w_kv(kv).reshape(b, n, 2, h, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q.reshape(b * h, n, c // h)
        k = k.reshape(b * h, n, c // h)
        v = v.reshape(b * h, n, c // h)
        x = self.adaptive_cluster_attention(q, k, v, cluster, self.attention_dropout)
        x = x.reshape(b, h, n, c // h)

        x = x.transpose(1, 2).reshape(b, n, c)  # (b, h, n, e) -> (b, n, h, e) -> (b, n, c)

        x = self.proj(x)
        x = self.projection_dropout(x)
        return x

    def _create_clusters(self, x, clusters):
        """
        count the number of clusters as weights
        """
        b, n, d = x.shape

        groups = clusters.int()
        if n == groups.size(-1) + 1:
            # original implementation: x contains cls token
            c = self.number_clusters + 1
        else:
            c = self.number_clusters

        counts = torch.zeros((clusters.shape[0], c), dtype=torch.int, device=clusters.device)
        for i in range(0, groups.shape[0]):
            counts[i, :] = groups[i, :].bincount(minlength=c)
        groups = groups.repeat(int(b / clusters.shape[0]), 1)
        counts = counts.repeat(int(b / clusters.shape[0]), 1)

        return groups, counts.contiguous()

    def adaptive_cluster_attention(self, q, k, v, cluster, attention_dropout=None):
        if self.autograd.requires_grad:
            # pytorch lightning use autograd
            # if you don't want to calculate gradient, you can use inf.
            cal_center_func = CalcCenter.apply
            broadcast_func = Broadcast.apply
        else:
            cal_center_func = CalcCenter.inf
            broadcast_func = Broadcast.inf

        if self.group_q:
            q_groups, q_counts = self._create_clusters(q, cluster)
            q_center = cal_center_func(q, q_groups, q_counts)
        else:
            q_groups = None
            q_center = q

        if self.group_k:
            k_groups, k_counts = self._create_clusters(k, cluster)
            k_center = cal_center_func(k, k_groups, k_counts)
            v_center = cal_center_func(v, k_groups, k_counts)
        else:
            k_counts = None
            k_center = k
            v_center = v

        # original implementation has no sqrt(dim_k), contained in the softmax_temp
        dim_k = torch.tensor(k_center.size(-1), dtype=torch.float, device=k_center.device)
        attention_score = torch.bmm(q_center, k_center.transpose(1, 2)) / torch.sqrt(dim_k)

        if attention_dropout is not None:
            attention_score = attention_dropout(attention_score)
        # if k_counts = 0, then attention_weight = 0 (prevent inf)
        attention_weight = self.softmax(self.softmax_temp * attention_score, dim=-1, weight=k_counts)
        output = torch.bmm(attention_weight, v_center)

        if self.group_q:  # why broadcast q_center?
            output = broadcast_func(output, q_groups)
        return output
