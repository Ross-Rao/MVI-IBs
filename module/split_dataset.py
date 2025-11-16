# python import
import os
import logging
from typing import Union
import shutil
# package import
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
# local import

__all__ = ["split_dataset_folds_and_save"]

logger = logging.getLogger(__name__)


def split_dataset_folds_and_save(
        df: pd.DataFrame,
        n_folds: int,
        test_split_ratio: float,
        save_dir: str,
        split_cols: Union[list, None] = None,
        shuffle: bool = True,
        seed: int = 42,
        use_existing_split: bool = False,
        reset_split_index: bool = True,
        train_file_name: str = "train_{0}.csv",
        val_file_name: str = "val_{0}.csv",
        test_file_name: str = "test.csv",
):
    paths = [os.path.join(save_dir, test_file_name)]
    paths += [os.path.join(save_dir, train_file_name.format(fold)) for fold in range(n_folds)]
    paths += [os.path.join(save_dir, val_file_name.format(fold)) for fold in range(n_folds)]
    if all([os.path.exists(path) for path in paths]):
        if use_existing_split:
            logger.info(f"Dataset split already exists in {save_dir}. Skipping split.")
            return
        else:
            logger.info(f"Dataset split already exists in {save_dir}. Overwriting files.")

    # Check and create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Split test set
    if split_cols is not None and len(split_cols) > 0:
        group_keys = df.groupby(split_cols).groups.keys()
        group_keys = [key if isinstance(key, tuple) else (key,) for key in group_keys]
        assert len(group_keys) > n_folds, "Number of unique groups must be greater than number of folds."
        train_keys, test_keys = train_test_split(group_keys, test_size=test_split_ratio,
                                                 shuffle=shuffle, random_state=seed)
        train_val_df = df[df[split_cols].apply(tuple, axis=1).isin(train_keys)]
        test_df = df[df[split_cols].apply(tuple, axis=1).isin(test_keys)]
    else:
        train_val_df, test_df = train_test_split(df, test_size=test_split_ratio,
                                                 shuffle=shuffle, random_state=seed)

    # Save test set
    if reset_split_index:
        test_df.reset_index(drop=True, inplace=True)
        test_df.index += len(train_val_df)
    test_df.to_csv(os.path.join(save_dir, test_file_name), index=True)

    # Split train and validation sets using KFold
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    if split_cols is not None and len(split_cols) > 0:
        train_val = train_val_df.groupby(split_cols).groups.keys()
        train_val = [key if isinstance(key, tuple) else (key,) for key in train_val]
    else:
        train_val = train_val_df

    for fold, (train_index, val_index) in enumerate(kf.split(train_val)):
        if split_cols is not None and len(split_cols) > 0:
            train_df = df[df[split_cols].apply(tuple, axis=1).isin([train_val[i] for i in train_index])]
            val_df = df[df[split_cols].apply(tuple, axis=1).isin([train_val[i] for i in val_index])]
        else:
            train_df = train_val.iloc[train_index]
            val_df = train_val.iloc[val_index]

        if reset_split_index:
            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            val_df.index += len(train_df)

        # Save train and validation sets
        train_df.to_csv(os.path.join(save_dir, train_file_name.format(fold)), index=True)
        val_df.to_csv(os.path.join(save_dir, val_file_name.format(fold)), index=True)

    logger.info(f"Dataset split completed. Files saved to {save_dir}")
    

if __name__ == "__main__":
    # example usage
    dataframe = pd.DataFrame({
        'file_path': [f'file_{i}.txt' for i in range(100)],
        'class': ['A'] * 50 + ['B'] * 50,
        'group': ['G1'] * 25 + ['G2'] * 25 + ['G1'] * 25 + ['G2'] * 25
    })
    print("Original DataFrame:")
    print(dataframe.head())

    example_dir = './data_splits'
    split_col = ['class', 'group']  # 替换为你的列名

    split_dataset_folds_and_save(
        dataframe,
        n_folds=3,
        test_split_ratio=0.2,
        save_dir=example_dir,
        split_cols=split_col,
        shuffle=True,
        seed=42,
        use_existing_split=False,
        reset_split_index=True
    )
    
    test_file = os.path.join(example_dir, "test.csv")
    train_file = os.path.join(example_dir, "train_0.csv")
    val_file = os.path.join(example_dir, "val_0.csv")

    # 读取文件
    test = pd.read_csv(test_file, index_col=0)
    train = pd.read_csv(train_file, index_col=0)
    val = pd.read_csv(val_file, index_col=0)

    # 获取每一行 split_col 的组合（元组），并去重
    test_values = set(tuple(row) for row in test[split_col].values)
    train_values = set(tuple(row) for row in train[split_col].values)
    val_values = set(tuple(row) for row in val[split_col].values)

    # 检查 test 中的值是否在 train 或 val 中出现
    for value in test_values:
        assert value not in train_values, f"{value} 出现在 train 集中"
        assert value not in val_values, f"{value} 出现在 val 集中"

    # 删除生成的文件夹
    shutil.rmtree(example_dir)
    