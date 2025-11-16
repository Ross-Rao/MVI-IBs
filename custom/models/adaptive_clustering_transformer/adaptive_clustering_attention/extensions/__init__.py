"""
@article{zheng2020end,
  title={End-to-end object detection with adaptive clustering transformer},
  author={Zheng, Minghang and Gao, Peng and Wang, Xiaogang and Li, Hongsheng and Dong, Hao},
  journal={arXiv preprint arXiv:2011.09315},
  year={2020}
}
"""
import torch
# local import: pylint: disable=import-error
if __package__ in [None, ""]:
    import cu_broadcast
    import cu_weighted_sum
else:
    from . import cu_broadcast
    from . import cu_weighted_sum


__all__ = ['broadcast', 'weighted_sum']


def broadcast(y, groups, weights):
    """
    Broadcasts the input tensor `y` to a new shape based on the `groups` and `weights`.

    Args:
        y (torch.Tensor): Input tensor of shape (b, c, d).
        groups (torch.Tensor): Tensor containing group indices of shape (b, n).
        weights (torch.Tensor): Tensor containing weights of shape (b, c).

    Returns:
        torch.Tensor: Broadcast tensor of shape (b, n, d).
    """
    assert y.size(1) == weights.size(1)
    b, c, d = y.shape
    n = groups.size(1)
    x = torch.zeros((b, n, d), dtype=torch.float, device=y.device)
    cu_broadcast.forward(y, groups, weights, x)
    return x


def weighted_sum(x, groups, weights):
    """
    Computes the weighted sum of the input tensor `x` based on the `groups` and `weights`.

    Args:
        x (torch.Tensor): Input tensor of shape (b, n, d).
        groups (torch.Tensor): Tensor containing group indices of shape (b, n).
        weights (torch.Tensor): Tensor containing weights of shape (b, c).

    Returns:
        torch.Tensor: Output tensor of shape (b, c, d) containing the weighted sum.
    """
    assert x.size(1) == groups.size(1)
    b, n, d = x.shape
    c = weights.size(1)
    y = torch.zeros((b, c, d), dtype=torch.float, device=x.device)
    cu_weighted_sum.forward(x, groups, weights, y)
    return y


def torch_weighted_sum(x, groups, weights):
    assert x.size(1) == groups.size(1)
    B, N, D = x.shape
    _, C = weights.shape

    y = torch.zeros((B, C, D), dtype=torch.float, device=x.device)
    # Iterate over batch dimension
    for b in range(B):
        for n in range(N):
            c_idx = groups[b, n]
            if c_idx >= 0 and c_idx < C:
                w = weights[b, c_idx]
                y[b, c_idx] += x[b, n] * w
    return y.contiguous()


def torch_broadcast(y, group, weights):
    B, C, D = y.shape
    _, N = groups.shape

    x = torch.zeros((B, N, D), dtype=torch.float, device=y.device)

    # Iterate over batch dimension
    for b in range(B):
        for n in range(N):
            c_idx = group[b, n]
            if c_idx >= 0 and c_idx < C:
                w = weights[b, c_idx]
                x[b, n] = y[b, c_idx] * w

    return x.contiguous()


if __name__ == '__main__':
    import torch
    import numpy as np

    b, n, c, d = 1024, 32, 128, 128
    y = torch.randn(b, n, d).cuda()
    y2 = torch.randn(b, c, d).cuda()
    groups = torch.randint(0, c, (b, n)).cuda()
    weights = torch.randn(b, c).cuda()

    x = broadcast(y2, groups, weights)
    x_torch = torch_broadcast(y2, groups, weights)

    assert torch.all(x == x_torch), ((x-x_torch).abs().max().item(),
                                     torch.mean((x == x_torch).float()).item(),
                                     (x != x_torch).sum().item())

    x = weighted_sum(y, groups, weights)
    x_torch = torch_weighted_sum(y, groups, weights)

    # if abs_max < 1e-6, then the two results are the same
    assert torch.all(x == x_torch), ((x-x_torch).abs().max().item(),
                                     torch.mean((x == x_torch).float()).item(),
                                     (x != x_torch).sum().item())
