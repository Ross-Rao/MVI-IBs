# python import
import os
import logging
from copy import deepcopy
from functools import partial
from typing import Union
# package import
import torch
import monai
import numpy as np
import lightning.pytorch as pl
# local import
from module.load_monai_dataset import load_monai_dataset

logger = logging.getLogger(__name__)
__all__ = ['MonaiDataModule']


class MonaiDataModule(pl.LightningDataModule):
    def __init__(self, dataset: dict[str, ...], data_loader: dict[str, ...], mixup: Union[None, dict[str, ...]]) -> None:
        super().__init__()

        self.dataset_config = dataset
        assert {'train_loader', 'val_loader', 'test_loader'}.issubset(data_loader.keys()), (
            "data_loader must contain 'train_loader', 'val_loader', and 'test_loader'")

        self.original_train_dataset, self.val_dataset, self.test_dataset = load_monai_dataset(**dataset)
        mixup = mixup if mixup is not None else {}
        mixup_ratio, mixup_keys = mixup.get('mixup_ratio', 0), mixup.get('mixup_keys', [])
        assert (mixup_ratio == 0) == (len(mixup_keys) == 0), \
            "mixup_ratio and mixup_keys must be both None or both not None, or you should check your mixup config."
        if mixup_ratio > 0 and mixup_keys is not None:
            self.train_dataset = self.mixup_dataset(self.original_train_dataset, keys=mixup_keys, ratio=mixup_ratio)
            logger.info(f"Applied mixup to training dataset with ratio {mixup_ratio} on keys {mixup_keys}. "
                        f"Original size: {len(self.original_train_dataset)}, New size: {len(self.train_dataset)}")
        else:
            self.train_dataset = self.original_train_dataset

        self.train_loader_params, self.val_loader_params, self.test_loader_params = (
            data_loader['train_loader'], data_loader['val_loader'], data_loader['test_loader'])

    def mixup_dataset(self, dataset, keys: list, ratio: float, alpha: float = 1.0, mix_package: str = 'numpy'):
        original_len = len(dataset)
        target_len = int(ratio * original_len)

        # 预生成随机混合比例
        if mix_package == "numpy":
            lambdas = torch.tensor(np.random.beta(alpha, alpha, size=target_len))
        elif mix_package == "torch":
            lambdas = torch.distributions.Beta(alpha, alpha).sample((target_len,))
        else:
            raise ValueError("mix_package must be 'numpy' or 'torch'")
        all_indices = torch.randint(0, original_len, (target_len, 2))
        indices_a , indices_b = all_indices[:, 0], all_indices[:, 1]

        mixed_data = []

        for index in range(target_len):
            sample_a = dataset[indices_a[index].item()]
            sample_b = dataset[indices_b[index].item()]
            lambda_val = lambdas[index].item()
            mixed_sample = deepcopy(sample_a)

            for key in keys:
                mixed_sample[key] = lambda_val * sample_a[key] + (1 - lambda_val) * sample_b[key]

            mixed_sample['mixup_index'] = index
            mixed_data.append(mixed_sample)
        if self.dataset_config.get('dataset_params') is None:
            self.dataset_config['dataset_params'] = {}
        dataset_class = partial(getattr(monai.data, self.dataset_config['dataset']),
                                **self.dataset_config.get('dataset_params'))
        return dataset_class(mixed_data, transform=None)

    def train_dataloader(self):
        return monai.data.DataLoader(self.train_dataset, **self.train_loader_params)

    def val_dataloader(self):
        return monai.data.DataLoader(self.val_dataset, **self.val_loader_params)

    def test_dataloader(self):
        return monai.data.DataLoader(self.test_dataset, **self.test_loader_params)


if __name__ == "__main__":
    import shutil
    import SimpleITK as sitk

    # 基础配置
    dataset_config = {
        "data_dir": "./example_data",
        "primary_key": "file_path",
        "parser": {
            "file_path": "lambda x: x.endswith('.nii.gz')",
            "patient_id": "lambda x: os.path.basename(x).split('_')[1]",
            "label": "lambda x: int(os.path.basename(x).split('_')[2].split('.')[0] == 'disease')",
        },
        "n_folds": 5,
        "fold": 0,
        "test_split_ratio": 0.2,
        "split_save_dir": "./example_data/split",
        "split_cols": ["patient_id"],
        "shuffle": True,
        "seed": 42,
        "use_existing_split": True,
        "reset_split_index": True,
        "transform": {
            "LoadImaged": {"keys": ["file_path"]},
            "ScaleIntensityD": {"keys": ["file_path"]},
            "ToTensorD": {"keys": ["file_path", "label"]},
            "AsDiscreteD": {"keys": ["label"], "to_onehot": 2},
        },
        "dataset": "Dataset",
    }

    # 1. 生成测试数据
    example_dir = dataset_config["data_dir"]
    if os.path.exists(example_dir):
        shutil.rmtree(example_dir)
    os.makedirs(example_dir)

    print("正在生成200个随机nii.gz文件...")
    for i in range(200):
        patient_id = f"patient_{i:03d}"
        label = np.random.choice(['healthy', 'disease'])
        filename = f"{patient_id}_{label}.nii.gz"

        random_data = np.random.randint(0, 255, (128, 128, 32), dtype=np.uint8)
        sitk_image = sitk.GetImageFromArray(random_data)
        sitk.WriteImage(sitk_image, os.path.join(example_dir, filename))

    print(f"已生成200个文件到 {example_dir}")

    loader_config = {
        'train_loader': {'batch_size': 8, 'shuffle': True, 'num_workers': 4},
        'val_loader': {'batch_size': 8, 'shuffle': False, 'num_workers': 4},
        'test_loader': {'batch_size': 8, 'shuffle': False, 'num_workers': 4},
    }

    mixup_config = {
        'mixup_ratio': 5,
        'mixup_keys': ['file_path', 'label']
    }

    data_module = MonaiDataModule(dataset=dataset_config, data_loader=loader_config, mixup=mixup_config)

    # 3. 测试各个 dataloader
    print("\n测试 DataLoaders:")

    # 测试训练集 dataloader
    train_loader = data_module.train_dataloader()
    print(f"训练集 DataLoader 创建成功, 数据集大小: {len(train_loader.dataset)}")

    # 测试验证集 dataloader
    val_loader = data_module.val_dataloader()
    print(f"验证集 DataLoader 创建成功, 数据集大小: {len(val_loader.dataset)}")

    # 测试测试集 dataloader
    test_loader = data_module.test_dataloader()
    print(f"测试集 DataLoader 创建成功, 数据集大小: {len(test_loader.dataset)}")

    # 4. 测试批次数据加载
    print("\n测试批次数据加载:")
    try:
        # 测试训练集第一个批次
        train_batch = next(iter(train_loader))
        print(f"训练集批次形状 - 图像: {train_batch['file_path'].shape}, 标签: {train_batch['label'].shape}")

        # 测试验证集第一个批次
        val_batch = next(iter(val_loader))
        print(f"验证集批次形状 - 图像: {val_batch['file_path'].shape}, 标签: {val_batch['label'].shape}")

        # 测试测试集第一个批次
        test_batch = next(iter(test_loader))
        print(f"测试集批次形状 - 图像: {test_batch['file_path'].shape}, 标签: {test_batch['label'].shape}")

        print("\n✅ MonaiDataModule 测试成功!")

    except Exception as e:
        print(f"\n❌ 批次数据加载失败: {e}")

    # 5. 清理生成的文件
    print("\n清理生成的文件...")
    shutil.rmtree(example_dir)
    if os.path.exists("./example_data/split"):
        shutil.rmtree("./example_data/split")
    print("测试完成")