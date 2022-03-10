# Created on February 19, 2022
# Author: Liang YuChao

# 导入python包 ################################################################
import os
import numpy as np
import pandas as pd
now_path = os.getcwd()
from joblib import dump
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
try:
    from . import Read
except:
    import Read

# 读取svm文件为numpy格式
def svm_to_numpy(file, rootPath=now_path):
    with open(file, 'r', encoding='utf-8') as f:
        content = f.readlines()
    # 提取特征list
    features = []
    features_label = []
    for i in content:
        line = i.strip('\n').split(' ')
        fs_box = line[1:]
        mid_box = []
        for j in fs_box:
            mid_box.append(float(j.split(':')[-1]))
        features.append(mid_box)
        features_label.append(int(line[0]))
    # 转换为数组
    np_data = np.array(features)
    np_label = np.array(features_label)
    return np_data, np_label

# 读取csv文件为numpy格式
def csv_to_numpy(file, rootPath):
    try:
        inputData = pd.read_csv(os.path.join(rootPath, file)) # 尝试全路径读取文件
    except:
        inputData = pd.read_csv(file) # 直接读取文件
    X_train = np.array(inputData.iloc[:,:-1]) # 获取数据数组
    y_train = np.array(inputData.iloc[:,-1]) # 获取标签数组
    labels = inputData.columns.values.tolist()[:-1]
    return X_train, y_train, labels

# 将numpy保存为csv文件
def numpy_to_csv(X_train, y_train, labels, outPath, rootPath):
    add_y = np.empty((len(y_train), 1))
    for j in range(len(y_train)):
        add_y[j][0] = y_train[j]
    all_data = np.concatenate((X_train, add_y),axis=1)
    all_dataframe = pd.DataFrame(all_data, columns=labels+['type'])
    all_dataframe.to_csv(outPath, index = False)
    try:
        all_dataframe.to_csv(os.path.join(rootPath, outPath), index = False)
        return os.path.join(rootPath, outPath)
    except:
        all_dataframe.to_csv(outPath, index = False)
        return outPath

# 数据归一化（创建模型）
def np_to_scale(X_train, saveName, rootPath):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    try:
        dump(scaler, os.path.join(rootPath, saveName))
    except:
        dump(scaler, saveName)
    return X_train

# 数据归一化（已有模型）
def np_to_scale_model(X_train, modelName, rootPath):
    try:
        scaler = load(os.path.join(rootPath, modelName))
    except:
        scaler = load(modelName)
    X_train = scaler.transform(X_train)
    return X_train

# 保存为svm格式特征文件
def np_to_svm(X_train, y_train, out):
    X_train = np_to_scale(X_train)
    content = ''
    for i in range(len(y_train)):
        line = str(y_train[i])
        for j in range(len(X_train[i])):
            line += ' ' + str(j+1) + ':' + str(X_train[i][j])
        content += line + '\n'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(content)
    return out

# 提取特定特征
def feature_to_fs(X_fs, X_train, i=0):
    add_x = np.empty((len(X_train), 1))
    for j in range(len(X_train)):
        add_x[j][0] = X_train[j][i]
    X_fs = np.concatenate((X_fs, add_x),axis=1)
    return X_fs

# 根据排序文件提取特征
def sort_to_feature(X_train, file_path='Analyze', number=10):
    with open(os.path.join(file_path, 'Fsort-pca.txt'), 'r', encoding='utf-8') as f:
        sort_list = f.read().split(': ')[-1].split(' ')[:-1]
    out_data = np.empty((len(X_train), 0))
    for i in range(number):
        out_data = Read.feature_to_fs(out_data, X_train, i=int(sort_list[i])-1)
    return out_data

# 读取数据并提取特征
def main(file, out, rootPath=now_path, sp=None):
    X_train, y_train, labels = csv_to_numpy(file, rootPath)
    if sp == None:
        out_path = np_to_svm(X_train, y_train, os.path.join(rootPath, out))
        return out_path
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=sp, random_state=123) # 分割数据
        out_path = np_to_svm(X_train, y_train, os.path.join(rootPath, 'train_'+out))
        out_path = np_to_svm(X_test, y_test, os.path.join(rootPath, 'test_'+out))
        return os.path.split(out_path)[0]
