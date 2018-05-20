
import os

import csv
import numpy as np

def csvpath2npdatas(train_path, valid_path, forChainer=False, forPytorch=False):
    train_datas = []
    train_targets = []
    valid_datas = []
    valid_targets = []
    ids = []
    test_datas = []
    live_datas = []
    eras = []

    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            train_datas.append(row[3:-1])
            train_targets.append(row[-1])

    with open(valid_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            ids.append(row[0])
            eras.append(row[1])
            if row[2]=='validation':
                valid_datas.append(row[3:-1])
                valid_targets.append(row[-1])
            elif row[2]=='test':
                test_datas.append(row[3:-1])
            else:
                live_datas.append(row[3:-1])

    if forChainer:
        train_datas = np.array(train_datas, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.int32)
        valid_datas = np.array(valid_datas, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.int32)
        test_datas = np.array(test_datas, dtype=np.float32)
        live_datas = np.array(live_datas, dtype=np.float32)
    if forPytorch:
        train_datas = np.array(train_datas, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.int64)
        valid_datas = np.array(valid_datas, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.int64)
        test_datas = np.array(test_datas, dtype=np.float32)
        live_datas = np.array(live_datas, dtype=np.float32)
    return train_datas, train_targets, valid_datas, valid_targets, ids, test_datas, live_datas, eras