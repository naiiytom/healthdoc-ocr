import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import time


def read_file(path):
    df = pd.read_csv(path, header=None)
    return df


def split_train_test(df, size=0.2):
    train, test = train_test_split(df, test_size=size, random_state=32)
    return train, test


def write_text_file(df, file_name):
    df.to_csv(file_name, index=False, header=False)


if __name__ == "__main__":
    start = time.time()
    print(f'Reading')
    file = read_file('D:/doc3d/listfile.txt')
    train, test = split_train_test(file)
    val, test = split_train_test(test, size=0.5)
    print(
        f'Train size: {len(train)} | Test size: {len(test)} | Val size: {len(val)}')
    print(f'Writing train list file')
    write_text_file(train, 'D:/doc3d/train.txt')
    print(f'Writing test list file')
    write_text_file(test, 'D:/doc3d/test.txt')
    print(f'Writing validation list file')
    write_text_file(val, 'D:/doc3d/val.txt')
    print(f'Finish in {(time.time() - start):.4f} seconds')
