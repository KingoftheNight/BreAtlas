# Created on February 19, 2022
# Author: Liang YuChao

# 导入python包 ################################################################
import os
import sys
import numpy as np
import pandas as pd
from skrebate import TuRF
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
try:
    from . import Read
    from . import Models
    from . import Plot
except:
    import Read
    import Models
    import Plot

# 进度显示
def visual_easy_time(start_e, end_e):
    print('\r>>>' + str(start_e) + "~" + str(end_e), end='', flush=True)

# 特征排序
def select_sort_rf(data):
    arr = []
    for i in data:
        arr.append(i)
    index = []
    for i in range(len(arr)):
        index.append(i)
    for i in range(len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        index[min_index], index[i] = index[i], index[min_index]
        arr[min_index], arr[i] = arr[i], arr[min_index]
    # 倒序输出
    re_index = []
    for i in range(len(index) - 1, -1, -1):
        re_index.append(index[i])
    return re_index

# 特征组合测试
def select_test(data, label, feature, now_path):
    fs_acc = []
    filter_data = []
    for k in label:
        filter_data.append(str(k))
    start_e = 0
    test_data, test_label = np.empty((len(data), 0)), label
    for i in range(len(feature)):
        start_e += 1
        test_data = Read.feature_to_fs(test_data, data, i=feature[i])
        test_label, predict_label = test_label, Models.MyKNN(test_data, test_label, cv=5, out=True)
        standard_num = Models.Performance(test_label, predict_label)
        single_acc = round(standard_num[4], 3)
        fs_acc.append(single_acc)
        visual_easy_time(start_e, len(feature))
    return fs_acc, test_data

# 保存排序结果
def select_save(out, fs_sort):
    out_file = 'IFS-feature-sort: '
    for j in fs_sort:
        out_file += str(j + 1) + ' '
    with open(out, 'w', encoding='UTF-8') as f:
        f.write(out_file)
        f.close()

# 保存筛选后特征文件
def select_result(out, labels, fs_sort, fs_importance, X_fs, y_train, number):
    sort_fs = np.zeros((len(fs_sort),1)).astype('int64')
    impo_fs = np.zeros((len(fs_importance),1)).astype('float64')
    for i in range(len(fs_sort)):
        sort_fs[i] = fs_sort[i]
        impo_fs[i] = fs_importance[i]
    out_data =np.concatenate((sort_fs, impo_fs),axis=1)
    out_label = []
    for i in range(number):
        out_label.append(labels[fs_sort[i]])
    out_df = pd.DataFrame(out_data, columns=['Index', 'Importance'])
    out_df.to_csv(out, encoding='utf-8')
    # X_train = np.array(out_df.iloc[:,:-1])
    # y_train = np.array(out_df.iloc[:,-1])
    # labels = out_df.columns.values.tolist()[:-1]
    return X_fs[:,:number], y_train, out_label

# 读取特征文件并进行分析返回最佳特征集
def select(X_train, y_train, labels, out_path='Analyze'):
    if out_path not in os.listdir(now_path):
        os.makedirs(out_path)
    # TuRF
    fs = TuRF(core_algorithm="ReliefF", n_features_to_select=2, pct=0.5,verbose=True)
    fs.fit(X_train, y_train, labels)
    fs_importance = fs.feature_importances_
    fs_sort = select_sort_rf(fs_importance)
    # get filter sort result
    fs_acc, X_fs = select_test(X_train, y_train, fs_sort, now_path)
    print('\n特征筛选完成，导出结果中...')
    # plot
    Plot.plot_ifs(fs_acc, 'IFS-Acc', os.path.join(out_path, 'Fsort-pca.png'))
    # sort file
    select_save(os.path.join(out_path, 'Fsort-pca.txt'), fs_sort)
    # feature selection numpy
    X_select, y_select, label_select = select_result(os.path.join(out_path, 'Fsort-data.csv'), labels, fs_sort, fs_importance, X_fs, y_train, fs_acc.index(max(fs_acc))+1)
    return X_select, y_select, label_select
