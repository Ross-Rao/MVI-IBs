# python import
# package import
import torch
import torch.nn as nn
# local import
if __package__ in [None, '']:
    from adaptive_clustering_attention import AdaptiveClusteringAttention
else:
    from .adaptive_clustering_attention import AdaptiveClusteringAttention


__all__ = ['AcTransformerEncoder']


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc. networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FeedForwardSequentialBlock(nn.Sequential):
    def __init__(self, dim, hidden_dim, output_dim=None, activation='GELU', dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            getattr(nn, activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim if output_dim is None else output_dim),
        )


class AcTransformerEncoderLayer(nn.Module):
    """
    Example:
        >>> import torch
        >>> tensor = torch.randn(1, 32, 512).cuda()
        >>> cluster = torch.randint(0, 128, (1, 32)).cuda()
        >>> model = AcTransformerEncoderLayer(
        ...     d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1
        ... ).to(tensor.device)
        >>> output = model(tensor, cluster)
        >>> print(output.shape)
        torch.Size([1, 32, 512])
        >>> loss = output.sum()
        >>> loss.backward()
    """
    def __init__(self, d_model,
                 num_heads,
                 output_dim=None,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='GELU',
                 attention_dropout=0.1,
                 drop_path_rate=0.1,
                 group_q=False,
                 group_k=True,
                 softmax_temp=1,
                 number_clusters=128
                 ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = AdaptiveClusteringAttention(d_model, num_heads, attention_dropout, dropout,
                                                group_q, group_k, softmax_temp, number_clusters)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForwardSequentialBlock(d_model, dim_feedforward, output_dim, activation, dropout)

    def forward(self, x, cluster):
        out1 = self.drop_path(self.attn(cluster, self.norm1(x)))
        out1 = x + out1
        out2 = self.drop_path(self.mlp(self.norm2(out1)))
        out2 = out1 + out2
        return out2


class AcTransformerEncoder(nn.Module):
    def __init__(self, n_layers: int = 12, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([AcTransformerEncoderLayer(**kwargs) for _ in range(n_layers)])

    def forward(self, x, cluster):
        for layer in self.layers:
            x = layer(x, cluster)
        return x


class TransformerSequentialClassificationHead(nn.Sequential):
    def __init__(self, d_model, num_classes):
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
