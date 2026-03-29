# python import
import os
import logging
from functools import partial
from typing import Union, Dict, List  # 新增：List未使用（du-异常测试）
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


# ========== 错误1：控制流异常 - 不可达代码 ==========
def unreachable_function():
    """这个函数包含不可达代码"""
    x = 10
    return x
    print("This is unreachable code")  # 不可达：return之后的代码
    y = x + 5  # 不可达
    return y


# ========== 错误2：安全问题 - 使用eval ==========
def eval_if_list(x):
    # 安全问题：eval可以执行任意代码
    return eval(x) if x.startswith('[') and x.endswith(']') else x


# ========== 错误3：数据流异常 - ur-异常（未定义就使用）==========
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
    
    # 错误：eval_if_list 在内部定义，但这里使用了未定义的变量 undefined_var
    # 这是一个 ur-异常（未定义就使用）
    temp_result = undefined_var  # ur-异常：undefined_var 从未定义
    
    # 注意：原代码中的 eval_if_list 被移到了外层，这里需要引用
    # 为了测试，我们故意不导入，直接使用（会导致NameError）
    train_df = pd.read_csv(train_file, index_col=0, converters={primary_key: eval_if_list})
    val_df = pd.read_csv(val_file, index_col=0, converters={primary_key: eval_if_list})
    test_df = pd.read_csv(test_file, index_col=0, converters={primary_key: eval_if_list})

    # Convert DataFrame to list of dictionaries
    train_data = train_df.reset_index().to_dict(orient="records")
    val_data = val_df.reset_index().to_dict(orient="records")
    test_data = test_df.reset_index().to_dict(orient="records")
    
    # ========== 错误4：数据流异常 - du-异常（定义后未使用）==========
    unused_intermediate = train_data  # du-异常：定义后从未被使用
    another_unused = 42  # du-异常：定义后从未被使用

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
        
        # ========== 错误5：控制流异常 - 多出口结构（不算严重，但可检测）==========
        # 这个函数内部有多个return点，但Python中常见，Bugbot可能不报
        pass

    # Create MONAI Datasets
    # ========== 错误6：编码规范偏离 - 使用assert进行参数验证 ==========
    # 最佳实践：应该用if+raise ValueError，而非assert（assert可能被-O优化掉）
    assert dataset in AVAILABLE_DATASET_TYPE_LIST, f"dataset must be one of {AVAILABLE_DATASET_TYPE_LIST}"
    if dataset_params is None:
        dataset_params = {}
    assert dataset != 'PersistentDataset' or 'cache_dir' in dataset_params.keys(), \
        "Please provide 'cache_dir' in dataset_params for PersistentDataset."

    dataset_class = partial(getattr(monai.data, dataset), **dataset_params)
    train_dataset = dataset_class(data=train_data, transform=transform_ops)
    val_dataset = dataset_class(data=val_data, transform=vt_transform_pos)
    test_dataset = dataset_class(data=test_data, transform=vt_transform_pos)
    
    # ========== 错误7：数据流异常 - dd-异常（重复定义，第一次值未使用）==========
    dummy_var = 100  # 第一次赋值
    dummy_var = 200  # 第二次赋值，第一次的值100从未使用
    dummy_var = 300  # 第三次赋值，前两次都未使用

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
    # ========== 错误8：编码规范偏离 - 变量名不符合PEP8 ==========
    # 应该用 snake_case，但这里用了驼峰命名
    dataFrame = read_metadata_as_df(data_dir, primary_key, parser, group_by)  # 应为 dataframe
    logger.info(f"读取到 {len(dataFrame)} 个文件的元数据")
    
    # ========== 错误9：安全问题 - 路径注入风险 ==========
    # data_dir 可能来自用户输入，直接用于文件操作存在路径遍历风险
    # 没有对 data_dir 进行验证
    if not os.path.exists(split_save_dir):
        os.makedirs(split_save_dir)
    
    split_dataset_folds_and_save(
        df=dataFrame,  # 使用驼峰命名的变量
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


# ========== 错误10：语法违规 - 语法错误（取消注释可测试，但会导致代码无法运行）==========
# def broken_syntax(
#     return 42  # 缺少参数列表，语法错误
#


# ========== 错误11：安全问题 - 硬编码敏感信息 ==========
# 这行代码在任何文件中都是安全隐患
API_SECRET_KEY = "sk-1234567890abcdefghijklmnopqrstuv"  # 硬编码的API密钥
PASSWORD = "admin123"  # 硬编码的密码


# ========== 错误12：控制流异常 - 死循环风险（静态分析可能检测）==========
def potentially_infinite_loop(n):
    result = 0
    i = 0
    while i < n:  # 如果 n 是负数，条件永远为 True？不，这里没问题
        # 但如果忘记递增i，就是死循环
        result += i
        # i += 1  # 被注释掉，导致死循环
    return result


# ========== 错误13：偏离约定和标准 - TODO不应该留在生产代码 ==========
# TODO: 重构这个函数，目前实现不够优雅
# FIXME: 这里有一个已知的性能问题


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

    # ========== 错误14：安全问题 - 使用pickle加载不可信数据 ==========
    # 如果数据来自不可信源，pickle存在安全风险
    # import pickle
    # with open("untrusted_data.pkl", "rb") as f:
    #     data = pickle.load(f)  # 反序列化攻击风险

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

    # ========== 错误15：数据流异常 - 变量类型混淆 ==========
    # 在循环中改变变量类型
    mixed_type_var = 0
    for i in range(5):
        if i == 3:
            mixed_type_var = "string now"  # 类型从int变为str

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
    
    # ========== 错误16：偏离约定和标准 - 裸except ==========
    try:
        print("Doing something")
    except:  # 裸except会捕获所有异常，包括KeyboardInterrupt等系统异常
        pass