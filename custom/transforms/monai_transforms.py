# python import
# package import
import torch
import numpy as np
from monai.data import MetaTensor
from monai.transforms import MapTransform
# local import

__all__ = ['ExtractSlicesD']


class ExtractSlicesD(MapTransform):
    def __init__(self, keys, k, dim=0):
        super().__init__(keys)
        self.k = k
        self.dim = dim

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            original_type = type(image)

            # 将图像转换为 NumPy 数组进行处理
            if isinstance(image, MetaTensor) or isinstance(image, torch.Tensor):
                img_array = image.numpy()
            elif isinstance(image, np.ndarray):
                img_array = image
            else:
                raise ValueError(f"Unsupported input type: {original_type}")

            # 获取指定维度的切片数量
            num_slices = img_array.shape[self.dim]
            start_idx = num_slices // 2 - self.k // 2
            end_idx = start_idx + self.k

            # 确保索引在有效范围内
            start_idx = max(0, start_idx)
            end_idx = min(num_slices, end_idx)

            # 提取切片
            if original_type == np.ndarray:
                # 对于 ndarray，直接切片
                d[key] = np.take(img_array, np.arange(start_idx, end_idx), axis=self.dim)
            elif original_type == torch.Tensor:
                # 对于 PyTorch Tensor，使用 torch 的切片操作
                d[key] = torch.take(torch.from_numpy(img_array), torch.arange(start_idx, end_idx), dim=self.dim)
            elif original_type == MetaTensor:
                # 对于 MetaTensor，切片并保留元数据
                sliced_image = np.take(img_array, np.arange(start_idx, end_idx), axis=self.dim)
                d[key] = MetaTensor(sliced_image, meta=image.meta)
            else:
                raise ValueError(f"Unsupported data type: {original_type}")
        return d
