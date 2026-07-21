from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

def read_csv_time_series(
    csv_path: str,
    date_col: int = 0,
    dtype: torch.dtype = torch.float32,
    nrows: int | None = None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    要求：
    - 第一列是 date
    - 第一行是通道名称
    实现：
    - pandas header=0 自动把第一行当列名（不进入数据）
    - 跳过 date 列，仅保留真正的数值通道
    返回：
    - data: [T, C] tensor
    - channel_names: List[str], 长度 C
    """
    if nrows is not None and nrows < 0:
        raise ValueError("nrows must be nonnegative or None")
    df = pd.read_csv(csv_path, header=0, nrows=nrows)
    cols = list(df.columns)
    value_cols = [c for i, c in enumerate(cols) if i != date_col]

    channel_names = value_cols[:]  # 记录通道名称
    values = df[value_cols].to_numpy(dtype=np.float32)  # [T, C]
    data = torch.tensor(values, dtype=dtype)            # [T, C]
    return data, channel_names
