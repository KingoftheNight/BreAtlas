# Created on February 19, 2022
# Author: Liang YuChao

# 导入python包 ################################################################
import os
import sys
import time
from sklearn.model_selection import train_test_split
now_path = os.getcwd() # 读取当前运行路径，防止文件保存到其他位置
file_path = os.path.dirname(__file__) # 读取脚本地址，便于调用同目录下其他文件
sys.path.append(file_path) # 将脚本地址添加入调用路径，便于调用同级目录下其他脚本
try:
    from . import Read # 尝试从当前目录导入python包
    from . import Filter
    from . import Models
except:
    import Read # 导入python包
    import Filter
    import Models

inputFile = input('Please enter a data file name: ')

# 读取数据
print('\n>>>Reading file: '+inputFile)
time.sleep(1)
X_train, y_train, labels = Read.csv_to_numpy(inputFile, now_path) # 读取指定文件并返回数据
print('\n>>>Readed successfully!\n\n>>>input data size: ('+str(len(X_train))+', '+str(len(X_train[0]))+')\n\n>>>input labels: '+','.join(labels[:5])+'...')
time.sleep(2)

# 数据分割
print('\n>>>Spliting data')
time.sleep(1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=123)
print('\n>>>Splited successfully!\n\n>>>train data size: ('+str(len(X_train))+', '+str(len(X_train[0]))+')\n\n>>>test data size: ('+str(len(X_test))+', '+str(len(X_test[0]))+')')
time.sleep(2)

# 归一化处理
print('\n>>>Normalizing data')
time.sleep(1)
X_train, X_test = Read.np_to_scale(X_train, 'train_scale.model', now_path), Read.np_to_scale_model(X_test, 'train_scale.model', now_path)
print('\n>>>Data has been normalized, and normalization model has been saved as train_scale.model')
time.sleep(2)

# 五折交叉（原始）
print('\n>>>Training data')
time.sleep(1)
origin_model = Models.MyKNN(X_train, y_train, out='origin_train.model') # 训练数据得出模型
print('\n>>>Successful!\n\n>>>Model has been saved as origin_train.model')
time.sleep(1)
origin_performance = Models.MyKNN(X_train, y_train, cv=5) # 五折交叉验证
print('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_performance[0])+'\t'+str(origin_performance[1])+'\t'+str(origin_performance[2])+'\t'+str(origin_performance[3])+'\t'+str(round(origin_performance[4],4))+'\t'+str(round(origin_performance[5],4))+'\t'+str(round(origin_performance[6],4))+'\t'+str(round(origin_performance[7],4))+'\t'+str(round(origin_performance[8],4))+'\n\n>>>Five-fold cross-validation results has been saved as origin_train.model')
time.sleep(2)

# 独立测试（原始）
print('\n>>>Testing data')
time.sleep(1)
origin_self_performance = Models.MyKNN(X_test, y_test, model=origin_model)
print('\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_self_performance[0])+'\t'+str(origin_self_performance[1])+'\t'+str(origin_self_performance[2])+'\t'+str(origin_self_performance[3])+'\t'+str(round(origin_self_performance[4],4))+'\t'+str(round(origin_self_performance[5],4))+'\t'+str(round(origin_self_performance[6],4))+'\t'+str(round(origin_self_performance[7],4))+'\t'+str(round(origin_self_performance[8],4)))
time.sleep(1)

# 保存结果（原始）
with open(os.path.join(now_path, 'origin_process_result.txt'), 'w', encoding='utf-8') as f:
    f.write('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_performance[0])+'\t'+str(origin_performance[1])+'\t'+str(origin_performance[2])+'\t'+str(origin_performance[3])+'\t'+str(round(origin_performance[4],4))+'\t'+str(round(origin_performance[5],4))+'\t'+str(round(origin_performance[6],4))+'\t'+str(round(origin_performance[7],4))+'\t'+str(round(origin_performance[8],4))+'\n\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_self_performance[0])+'\t'+str(origin_self_performance[1])+'\t'+str(origin_self_performance[2])+'\t'+str(origin_self_performance[3])+'\t'+str(round(origin_self_performance[4],4))+'\t'+str(round(origin_self_performance[5],4))+'\t'+str(round(origin_self_performance[6],4))+'\t'+str(round(origin_self_performance[7],4))+'\t'+str(round(origin_self_performance[8],4)))
print('\n>>>Result has been saved as origin_process_result.txt')

# 特征分析
print('\n>>>Selecting features')
time.sleep(1)
X_train_fs, y_train_fs, labels_fs = Filter.select(X_train, y_train, labels, out_path='Analyze')
X_test_fs = Read.sort_to_feature(X_test, file_path='Analyze', number=len(labels_fs))
print('\n>>>Selected successfully!\n\n>>>best-n-features: '+'|'.join(labels_fs)+' ('+str(len(labels_fs))+')\n\n>>>selected train data size: ('+str(len(X_train_fs))+', '+str(len(X_train_fs[0]))+')\n\n>>>selected test data size: ('+str(len(X_test_fs))+', '+str(len(X_test_fs[0]))+')\n\n>>>Selection result has been saved in Analyze folder')
time.sleep(2)

# 五折交叉（筛选）
print('\n>>>Training data')
time.sleep(1)
model = Models.MyKNN(X_train_fs, y_train_fs, out='train.model') # 训练数据得出模型
print('\n>>>Successful!\n\n>>>Model has been saved as train.model')
time.sleep(1)
performance = Models.MyKNN(X_train_fs, y_train_fs, cv=5) # 五折交叉验证
print('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(performance[0])+'\t'+str(performance[1])+'\t'+str(performance[2])+'\t'+str(performance[3])+'\t'+str(round(performance[4],4))+'\t'+str(round(performance[5],4))+'\t'+str(round(performance[6],4))+'\t'+str(round(performance[7],4))+'\t'+str(round(performance[8],4))+'\n\n>>>Five-fold cross-validation results has been saved as train.model')
time.sleep(2)

# 独立测试（筛选）
print('\n>>>Testing data')
time.sleep(1)
self_performance = Models.MyKNN(X_test_fs, y_test, model=model)
print('\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(self_performance[0])+'\t'+str(self_performance[1])+'\t'+str(self_performance[2])+'\t'+str(self_performance[3])+'\t'+str(round(self_performance[4],4))+'\t'+str(round(self_performance[5],4))+'\t'+str(round(self_performance[6],4))+'\t'+str(round(self_performance[7],4))+'\t'+str(round(self_performance[8],4)))
time.sleep(1)

# 保存结果（筛选）
with open(os.path.join(now_path, 'process_result.txt'), 'w', encoding='utf-8') as f:
    f.write('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(performance[0])+'\t'+str(performance[1])+'\t'+str(performance[2])+'\t'+str(performance[3])+'\t'+str(round(performance[4],4))+'\t'+str(round(performance[5],4))+'\t'+str(round(performance[6],4))+'\t'+str(round(performance[7],4))+'\t'+str(round(performance[8],4))+'\n\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(self_performance[0])+'\t'+str(self_performance[1])+'\t'+str(self_performance[2])+'\t'+str(self_performance[3])+'\t'+str(round(self_performance[4],4))+'\t'+str(round(self_performance[5],4))+'\t'+str(round(self_performance[6],4))+'\t'+str(round(self_performance[7],4))+'\t'+str(round(self_performance[8],4)))
print('\n>>>Result has been saved as process_result.txt')
