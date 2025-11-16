# python import
import os
import logging
from typing import Dict, Union
# package import
import pandas as pd
# local import

__all__ = ['read_metadata_as_df']

logger = logging.getLogger(__name__)


def read_metadata_as_df(data_dir: str, primary_key: str, parser: Dict[str, str], group_by: Union[list[str], None] = None):
    assert primary_key in parser.keys(), (f'Please add {primary_key} in your parser, using a lambda function to choose '
                                          f'which file is needed in your folder: {data_dir}.')
    eval_parser = {k: eval(v) for k, v in parser.items()}
        
    metadata = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_metadata = {}
            input_path = os.path.join(root, file)

            # parse image files, invalid file skipped
            if not eval_parser[primary_key](input_path):
                continue
            else:
                file_metadata[primary_key] = input_path

            # parse metadata
            for key, parser in eval_parser.items():
                if key == primary_key:
                    continue
                elif not callable(parser):
                    raise ValueError(f"Parser for {key} should be a lambda function.")
                file_metadata[key] = parser(input_path)
            metadata.append(file_metadata)

    assert len(metadata) > 0, f"No valid files found in {data_dir}, please check the path or the function"
    logger.info(f"Found {len(metadata)} valid files in {data_dir}")
    meta_df = pd.DataFrame(metadata)

    if group_by:
        assert all([col in meta_df.columns for col in group_by]), f"Columns {group_by} not found in metadata"
        assert set(group_by) != set(meta_df.columns), "group_by columns should not be all columns"
        assert primary_key not in group_by, "image column should not be in group_by columns"

        meta_df = meta_df.groupby(group_by)[primary_key].apply(list).reset_index()
        logger.info(f"Grouped metadata by {group_by}")
    return meta_df

if __name__ == "__main__":
    # example usage
    example_dir = "."
    data_col_name = "py_files"
    function_parser = {
        "py_files": "lambda x: x.endswith('.py')",
        "name": "lambda x: os.path.basename(x).split('.')[0]",
        'module': "lambda path: os.path.dirname(path)",
    }

    df = read_metadata_as_df(example_dir, data_col_name, function_parser, None)
    print(df.head())
    
    df = read_metadata_as_df(example_dir, data_col_name, function_parser, ["module"])
    print(df.head())
    