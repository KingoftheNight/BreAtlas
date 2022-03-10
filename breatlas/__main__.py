# Created on February 19, 2022
# Author: Liang YuChao

# 导入python包 ################################################################
import os
import sys
import argparse
from sklearn.model_selection import train_test_split
now_path = os.getcwd()  # 读取当前运行路径，防止文件保存到其他位置
file_path = os.path.dirname(__file__)  # 读取脚本地址，便于调用同目录下其他文件
sys.path.append(file_path)  # 将脚本地址添加入调用路径，便于调用同级目录下其他脚本
try:
    from . import Read  # 尝试从当前目录导入python包
    from . import Filter
    from . import Models
except:
    import Read  # 导入python包
    import Filter
    import Models

# 定义函数 ####################################################################


def function_split(args):
    print('>>>start parsing the file: '+args.file[0])
    X_train, y_train, labels = Read.csv_to_numpy(args.file[0], now_path)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=float(args.size[0]), random_state=123)
    outPath = Read.numpy_to_csv(
        X_train, y_train, labels, args.out[0].split('.')[0]+'-1.csv', now_path)
    print('>>>the train feature file has been saved in: '+outPath)
    outPath = Read.numpy_to_csv(
        X_test, y_test, labels, args.out[0].split('.')[0]+'-2.csv', now_path)
    print('>>>the test feature file has been saved in: '+outPath)


def function_scale(args):
    print('>>>start scaling the file: '+args.file[0])
    X_train, y_train, labels = Read.csv_to_numpy(args.file[0], now_path)
    if args.model:
        X_train = Read.np_to_scale_model(X_train, args.model[0], now_path)
    else:
        X_train = Read.np_to_scale(X_train, 'scale.model', now_path)
        print('>>>the scale model has been saved in: ' +
              os.path.join(now_path, 'scale.model'))
    outPath = Read.numpy_to_csv(X_train, y_train, labels, args.out[0], now_path)
    print('>>>the test feature file has been saved in: '+outPath)


def function_select(args):
    print('>>>start selecting the file: '+args.train[0])
    X_train, y_train, labels = Read.csv_to_numpy(args.train[0], now_path)
    X_test, y_test, labels = Read.csv_to_numpy(args.valid[0], now_path)
    X_train_fs, y_train_fs, labels_fs = Filter.select(X_train, y_train, labels, out_path=args.out[0])
    X_test_fs = Read.sort_to_feature(X_test, file_path=args.out[0], number=len(labels_fs))
    print('\n>>>Selected successfully!\n\n>>>best-n-features: '+'|'.join(labels_fs)+' ('+str(len(labels_fs))+')\n\n>>>selected train data size: ('+str(len(X_train_fs))+', '+str(len(X_train_fs[0]))+')\n\n>>>selected test data size: ('+str(len(X_test_fs))+', '+str(len(X_test_fs[0]))+')\n\n>>>Selection result has been saved in Analyze folder')
    outPath = Read.numpy_to_csv(X_train_fs, y_train_fs, labels_fs, args.out[0]+'-train.csv', now_path)
    print('>>>the selected train feature file has been saved in: '+outPath)
    outPath = Read.numpy_to_csv(X_test_fs, y_test, labels_fs, args.out[0]+'-valid.csv', now_path)
    print('>>>the selected valid feature file has been saved in: '+outPath)


def function_cvtest(args):
    print('>>>start cvtesting the file: '+args.file[0])
    X_train, y_train, labels = Read.csv_to_numpy(args.file[0], now_path)
    origin_model = Models.MyKNN(X_train, y_train, out=args.out[0])
    if args.crossv:
        origin_performance = Models.MyKNN(X_train, y_train, cv=int(args.crossv[0]))
    else:
        origin_performance = Models.MyKNN(X_train, y_train, cv=5)
    print('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_performance[0])+'\t'+str(origin_performance[1])+'\t'+str(origin_performance[2])+'\t'+str(origin_performance[3])+'\t'+str(round(origin_performance[4],4))+'\t'+str(round(origin_performance[5],4))+'\t'+str(round(origin_performance[6],4))+'\t'+str(round(origin_performance[7],4))+'\t'+str(round(origin_performance[8],4)))
    print(origin_model)
    if args.save:
        with open(os.path.join(now_path, args.save[0]), 'w', encoding='utf-8') as f:
            f.write('\n>>>Five-fold cross-validation results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_performance[0])+'\t'+str(origin_performance[1])+'\t'+str(origin_performance[2])+'\t'+str(origin_performance[3])+'\t'+str(round(origin_performance[4],4))+'\t'+str(round(origin_performance[5],4))+'\t'+str(round(origin_performance[6],4))+'\t'+str(round(origin_performance[7],4))+'\t'+str(round(origin_performance[8],4)))
        print('>>>the test result has been saved in: '+args.save[0])
    

def function_sftest(args):
    print('>>>start sftesting the file: '+args.file[0])
    X_test, y_test, labels = Read.csv_to_numpy(args.file[0], now_path)
    origin_self_performance = Models.MyKNN(X_test, y_test, model=args.model[0])
    print('\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_self_performance[0])+'\t'+str(origin_self_performance[1])+'\t'+str(origin_self_performance[2])+'\t'+str(origin_self_performance[3])+'\t'+str(round(origin_self_performance[4],4))+'\t'+str(round(origin_self_performance[5],4))+'\t'+str(round(origin_self_performance[6],4))+'\t'+str(round(origin_self_performance[7],4))+'\t'+str(round(origin_self_performance[8],4)))
    if args.save:
        with open(os.path.join(now_path, args.save[0]), 'w', encoding='utf-8') as f:
            f.write('\n>>>Self test results:\n\n\tTP\tTN\tFP\tFN\tACC\tSN\tSP\tPPV\tMCC\n\n\t'+str(origin_self_performance[0])+'\t'+str(origin_self_performance[1])+'\t'+str(origin_self_performance[2])+'\t'+str(origin_self_performance[3])+'\t'+str(round(origin_self_performance[4],4))+'\t'+str(round(origin_self_performance[5],4))+'\t'+str(round(origin_self_performance[6],4))+'\t'+str(round(origin_self_performance[7],4))+'\t'+str(round(origin_self_performance[8],4)))
        print('>>>the test result has been saved in: '+args.save[0])
        

# 参数解析 ####################################################################
def main():
    parser = argparse.ArgumentParser(
        description='Blood routine examination atlas.')  # 创建解析器对象
    subparsers = parser.add_subparsers(help='IRAP help') # 创建子解释器对象
    # split
    ParserSplit = subparsers.add_parser('split', help='Split csv file.')  # 创建子功能
    ParserSplit.add_argument('-f', '--file', nargs=1, help='input file name')  # 添加参数
    ParserSplit.add_argument('-o', '--out', nargs=1, help='output file name')
    ParserSplit.add_argument('-s', '--size', nargs=1, help='split ratio')
    ParserSplit.set_defaults(func=function_split)  # 绑定函数
    # scale
    ParserScale = subparsers.add_parser('scale', help='Scale file.')
    ParserScale.add_argument('-f', '--file', nargs=1, help='input file name')
    ParserScale.add_argument('-o', '--out', nargs=1, help='output file name')
    ParserScale.add_argument('-m', '--model', nargs=1, help='scale model')
    ParserScale.set_defaults(func=function_scale)
    # select
    ParserSelect = subparsers.add_parser('select', help='select feature and output analize file.')
    ParserSelect.add_argument('-t', '--train', nargs=1, help='input file name')
    ParserSelect.add_argument('-v', '--valid', nargs=1, help='valid file name')
    ParserSelect.add_argument('-o', '--out', nargs=1, help='output folder name')
    ParserSelect.set_defaults(func=function_select)
    # cvtest
    ParserCVtest = subparsers.add_parser('cvtest', help='cross-validation test.')
    ParserCVtest.add_argument('-f', '--file', nargs=1, help='input file name')
    ParserCVtest.add_argument('-o', '--out', nargs=1, help='output file name')
    ParserCVtest.add_argument('-cv', '--crossv', nargs=1, help='cross-validation')
    ParserCVtest.add_argument('-s', '--save', nargs=1, help='save result')
    ParserCVtest.set_defaults(func=function_cvtest)
    # sftest
    ParserSFtest = subparsers.add_parser('sftest', help='self test.')
    ParserSFtest.add_argument('-f', '--file', nargs=1, help='input file name')
    ParserSFtest.add_argument('-m', '--model', nargs=1, help='output file name')
    ParserSFtest.add_argument('-s', '--save', nargs=1, help='save result')
    ParserSFtest.set_defaults(func=function_sftest)

    # 解析参数
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        pass


# main
if __name__ == '__main__':
    main()
