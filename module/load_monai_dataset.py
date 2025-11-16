# python import
import os
import logging
from functools import partial
from typing import Union, Dict
# package import
import monai
import pandas as pd
from monai import transforms as monai_transforms
# local import
from custom import transforms as custom_transforms
from utils.load_module import get_unique_attr_across
from module.read_metadata import read_metadata_as_df
from module.split_dataset import split_dataset_folds_and_save

__all__ = ["load_monai_dataset"]

AVAILABLE_DATASET_TYPE_LIST = ['Dataset', 'CacheDataset', 'SmartCacheDataset', 'PersistentDataset']
logger = logging.getLogger(__name__)


def load_data_from_split_to_monai_dataset(
        load_dir: str,
        primary_key: str,
        fold: int,
        transform: Union[dict[str, ...], None] = None,
        val_test_transform: Union[dict[str, ...], None] = None,
        dataset: str = 'Dataset',
        dataset_params: Union[dict[str, str], None] = None,
        train_file_name: str = "train_{0}.csv",
        val_file_name: str = "val_{0}.csv",
        test_file_name: str = "test.csv",
):
    # Load train and validation datasets
    train_file = os.path.join(load_dir, train_file_name.format(fold))
    val_file = os.path.join(load_dir, val_file_name.format(fold))
    test_file = os.path.join(load_dir, test_file_name)
    
    def eval_if_list(x):
        return eval(x) if x.startswith('[') and x.endswith(']') else x

    train_df = pd.read_csv(train_file, index_col=0, converters={primary_key: eval_if_list})
    val_df = pd.read_csv(val_file, index_col=0, converters={primary_key: eval_if_list})
    test_df = pd.read_csv(test_file, index_col=0, converters={primary_key: eval_if_list})

    # Convert DataFrame to list of dictionaries
    train_data = train_df.reset_index().to_dict(orient="records")
    val_data = val_df.reset_index().to_dict(orient="records")
    test_data = test_df.reset_index().to_dict(orient="records")

    # Transform settings for train dataset
    if transform is None:
        transform_ops = monai_transforms.Compose([monai_transforms.ToTensor()])
    else:
        # if you want to use your own transform, you can add them to utils/custom_transforms.py
        # they will be imported by get_unique_attr_across
        transforms_lt = get_unique_attr_across([custom_transforms, monai_transforms, monai.data], transform)
        transform_ops = monai_transforms.Compose(transforms_lt)

    # Transform settings for val and test dataset
    if val_test_transform is None:
        vt_transform_pos = transform_ops
    else:
        transforms_lt = get_unique_attr_across([custom_transforms, monai_transforms, monai.data], val_test_transform)
        vt_transform_pos = monai_transforms.Compose(transforms_lt)

    # Create MONAI Datasets
    assert dataset in AVAILABLE_DATASET_TYPE_LIST, f"dataset must be one of {AVAILABLE_DATASET_TYPE_LIST}"
    if dataset_params is None:
        dataset_params = {}
    assert dataset != 'PersistentDataset' or 'cache_dir' in dataset_params.keys(), \
        "Please provide 'cache_dir' in dataset_params for PersistentDataset."

    dataset_class = partial(getattr(monai.data, dataset), **dataset_params)
    train_dataset = dataset_class(data=train_data, transform=transform_ops)
    val_dataset = dataset_class(data=val_data, transform=vt_transform_pos)
    test_dataset = dataset_class(data=test_data, transform=vt_transform_pos)

    return train_dataset, val_dataset, test_dataset


def load_monai_dataset(
    data_dir: str,
    primary_key: str,
    parser: Dict[str, str],
    n_folds: int,
    fold: int,
    test_split_ratio: float,
    split_save_dir: str,
    group_by: Union[list[str], None] = None,
    split_cols: Union[list, None] = None,
    shuffle: bool = True,
    seed: int = 42,
    use_existing_split: bool = False,
    reset_split_index: bool = True,
    transform: Union[dict[str, ...], None] = None,
    val_test_transform: Union[dict[str, ...], None] = None,
    dataset: str = 'Dataset',
    dataset_params: Union[dict[str, str], None] = None,
    train_file_name: str = "train_{0}.csv",
    val_file_name: str = "val_{0}.csv",
    test_file_name: str = "test.csv",
):
    dataframe = read_metadata_as_df(data_dir, primary_key, parser, group_by)
    logger.info(f"读取到 {len(dataframe)} 个文件的元数据")
    split_dataset_folds_and_save(
        df=dataframe,
        n_folds=n_folds,
        test_split_ratio=test_split_ratio,
        save_dir=split_save_dir,
        split_cols=split_cols,
        shuffle=shuffle,
        seed=seed,
        use_existing_split=use_existing_split,
        reset_split_index=reset_split_index,
        train_file_name=train_file_name,
        val_file_name=val_file_name,
        test_file_name=test_file_name,
    )

    train_dataset, val_dataset, test_dataset = load_data_from_split_to_monai_dataset(
        load_dir=split_save_dir,
        primary_key=primary_key,
        fold=fold,
        transform=transform,
        val_test_transform=val_test_transform,
        dataset=dataset,
        dataset_params=dataset_params,
        train_file_name=train_file_name,
        val_file_name=val_file_name,
        test_file_name=test_file_name,
    )
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    import shutil
    import numpy as np
    import SimpleITK as sitk

    # 基础配置
    base_config = {
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
            # "EnsureChannelFirstD": {"keys": ["file_path"]},  # 3d / 4d
            "ScaleIntensityD": {"keys": ["file_path"]},
            "ToTensorD": {"keys": ["file_path", "label"]},
        },
    }

    # 1. 生成测试数据
    example_dir = base_config["data_dir"]
    if os.path.exists(example_dir):
        shutil.rmtree(example_dir)
    os.makedirs(example_dir)

    print("正在生成200个随机nii.gz文件...")
    for i in range(200):
        patient_id = f"patient_{i:03d}"
        label = np.random.choice(['healthy', 'disease'])
        filename = f"{patient_id}_{label}.nii.gz"

        random_data = np.random.randint(0, 255, (64, 64, 32), dtype=np.uint8)
        sitk_image = sitk.GetImageFromArray(random_data)
        sitk.WriteImage(sitk_image, os.path.join(example_dir, filename))

    print(f"已生成200个文件到 {example_dir}")

    # 2. 测试不同数据集类型
    dataset_configs = [
        {"dataset": "Dataset", "dataset_params": None},
        {"dataset": "CacheDataset", "dataset_params": None},
        {"dataset": "SmartCacheDataset", "dataset_params": None},
        {"dataset": "PersistentDataset", "dataset_params": {"cache_dir": "./cache_dir"}},
    ]

    for config in dataset_configs:
        print(f"\n测试 {config['dataset']}:")
        try:
            # 合并基础配置和数据集特定配置
            test_config = {**base_config, **config}

            train_ds, val_ds, test_ds = load_monai_dataset(**test_config)

            print(f"  训练集样本数: {len(train_ds)}")
            print(f"  验证集样本数: {len(val_ds)}")
            print(f"  测试集样本数: {len(test_ds)}")

            # 测试加载一个样本
            sample = train_ds[0]
            print(f"  样本键: {list(sample.keys())}")
            print(f"  样本值: {list(sample.values())}")

        except Exception as e:
            print(f"  {config['dataset']} 测试失败: {e}")

    # 3. 清理生成的文件
    print("\n清理生成的文件...")
    shutil.rmtree(example_dir)
    if os.path.exists("./example_data/split"):
        shutil.rmtree("./example_data/split")
    if os.path.exists("./cache_dir"):
        shutil.rmtree("./cache_dir")
    print("测试完成")


