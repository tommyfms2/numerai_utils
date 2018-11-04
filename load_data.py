
import os

import csv
import numpy as np

def csvpath2npdatas(train_path, valid_path, forChainer=False, forPytorch=False):
    train_datas = []
    train_targets = [[],[],[],[],[]]
    valid_datas = []
    valid_targets = [[],[],[],[],[]]
    ids = []
    test_datas = []
    live_datas = []
    eras = []

    with open(train_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            train_datas.append(row[3:-5])
            train_targets[0].append(row[-5])
            train_targets[1].append(row[-4])
            train_targets[2].append(row[-3])
            train_targets[3].append(row[-2])
            train_targets[4].append(row[-1])

    with open(valid_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            ids.append(row[0])
            eras.append(row[1])
            if row[2]=='validation':
                valid_datas.append(row[3:-5])
                valid_targets[0].append(row[-5])
                valid_targets[1].append(row[-4])
                valid_targets[2].append(row[-3])
                valid_targets[3].append(row[-2])
                valid_targets[4].append(row[-1])

            elif row[2]=='test':
                test_datas.append(row[3:-5])
            else:
                live_datas.append(row[3:-5])

    if forChainer:
        train_datas = np.array(train_datas, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.int32)
        valid_datas = np.array(valid_datas, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.int32)
        test_datas = np.array(test_datas, dtype=np.float32)
        live_datas = np.array(live_datas, dtype=np.float32)
    if forPytorch:
        train_datas = np.array(train_datas, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.float32)
        train_targets = np.array(train_targets, dtype=np.int64)
        valid_datas = np.array(valid_datas, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.float32)
        valid_targets = np.array(valid_targets, dtype=np.int64)
        test_datas = np.array(test_datas, dtype=np.float32)
        live_datas = np.array(live_datas, dtype=np.float32)
    
    return train_datas, train_targets, valid_datas, valid_targets, ids, test_datas, live_datas, eras