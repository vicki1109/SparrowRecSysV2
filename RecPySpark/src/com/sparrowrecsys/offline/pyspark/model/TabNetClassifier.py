import sys

import numpy as np
import torch

from tab_model import TabNetClassifier


def readFile(file):
    file = open(file)
    val_list = file.readlines()
    y_arr = []
    x_arr = []
    for line in val_list:
        arr = line.split('\t')
        y_arr.append(arr[0])
        x_arr.append(arr[1:-1])
    return np.array(x_arr), np.array(y_arr)

if __name__ == '__main__':
    train_path = sys.argv[1]
    valid_path = sys.argv[2]

    # train_path = 'D:\\train.txt'
    # valid_path = 'D:\\test.txt'

    X_train, Y_train = readFile(train_path)
    X_valid, Y_valid = readFile(valid_path)

    clf = TabNetClassifier()
    clf.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
    torch.save(clf.network, 'model.pth')

    print('end')