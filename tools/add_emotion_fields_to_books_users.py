#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'environment', 'books', 'users_generation', 'datasets')
FILE_NAME = 'users_600.csv'
PATH = os.path.join(BASE_DIR, FILE_NAME)

DEFAULT_SEED = 42


def main(seed: int = DEFAULT_SEED):
    if not os.path.exists(PATH):
        print(f"File not found: {PATH}")
        sys.exit(1)

    df = pd.read_csv(PATH)

    rng = np.random.default_rng(seed)
    # 如果列不存在则添加
    if 'activity_level' not in df.columns:
        df['activity_level'] = rng.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])
    if 'conformity_level' not in df.columns:
        df['conformity_level'] = rng.choice([1, 2, 3], size=len(df), p=[0.25, 0.5, 0.25])
    if 'diversity_level' not in df.columns:
        df['diversity_level'] = rng.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])

    # 将三列移动到结尾（不改变已有列顺序）
    other_cols = [c for c in df.columns if c not in ['activity_level', 'conformity_level', 'diversity_level']]
    df = df[other_cols + ['activity_level', 'conformity_level', 'diversity_level']]

    # 覆盖写回
    df.to_csv(PATH, index=False, quoting=1)
    print(f"Updated {PATH} with emotion fields.")


if __name__ == '__main__':
    seed = DEFAULT_SEED
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except Exception:
            pass
    main(seed)
