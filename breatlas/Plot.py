import io
from PIL import Image
import seaborn as sns
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from Models import svm_grid
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import matplotlib

# 绘图函数
def plot_CM(data, title, out):
    # 创建xy轴label
    xlabel, ylabel = ['Positive','Negative'], ['Positive','Negative']
    # 创建画布fig, 子图ax(分辨率=300)
    f, ax = plt.subplots(dpi=300)
    # 绘制热图(色系为YlGnBu, 线宽为0.1, 单元格正方体为True, 显示xy轴label)
    sns.heatmap(data, cmap='Blues', linewidths=0.1, square=True, xticklabels=True, yticklabels=True, annot=True, fmt='.20g')
    # 设置子图ax标题
    ax.set_title(title)
    # 设置子图xy轴标签
    ax.set_ylabel('Predictive value')
    ax.set_xlabel('Actual value')
    # 为xy轴添加label
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()

def plot_ROC(X_train, y_train, out):
    cv = StratifiedKFold(n_splits=6)
    c, g = svm_grid(X_train, y_train)
    classifier = svm.SVC(kernel='rbf', C=c, gamma=g, probability=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    for train, test in cv.split(X_train, y_train):
        probas_ = classifier.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()
    return mean_auc

def plot_multy_ROC(X_train, y_train, out):
    cv = StratifiedKFold(n_splits=6)
    c, g = svm_grid(X_train, y_train)
    models = {
        'KNN': sklearn.neighbors.KNeighborsClassifier(n_neighbors=14,p=1),
        'SVM': svm.SVC(kernel='rbf', C=c, gamma=g, probability=True),
        'XGBoost': XGBClassifier(eta=0.01, objective="binary:logistic", subsample=0.5, eval_metric="logloss", base_score=np.mean(y_train)),
        'RFC': RandomForestClassifier(random_state=0, max_depth=None, n_estimators=100, min_samples_split=2)
        }
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for key in models:
        classifier = models[key]
        all_y, all_prob = np.empty(shape=(0,)), np.empty(shape=(0,2))
        for train, test in cv.split(X_train, y_train):
            each_prob = classifier.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
            all_y = np.concatenate((all_y,y_train[test]),axis=0)
            all_prob = np.concatenate((all_prob,each_prob),axis=0)
        fpr, tpr, thresholds = roc_curve(all_y, all_prob[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1,
                 label='AUC for %s = %0.4f' % (key, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()
    return mean_auc

def plot_ifs(data, type_p, out):
    x = []
    y = []
    for i in range(len(data)):
        x.append(i + 1)
    for j in data:
        y.append(j)
    plt.figure()
    plt.plot(x, y, label='ACC')
    plt.xlabel("Feature Number")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_x = y.index(max(y))
    max_y = max(y)
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()

def plot_multy_ifs(data, type_p, out):
    x = []
    for i in range(len(data[0])):
        x.append(i + 1)
    plt.figure()
    plt.plot(x, data[0], label='KNN', linewidth=1, c='#1f77b4')
    plt.plot(x, data[1], label='SVM', linewidth=1, c='#ff7f0e')
    plt.plot(x, data[2], label='XGBoost', linewidth=1, c='#2ca02c')
    plt.plot(x, data[3], label='RFC', linewidth=1, c='#d62728')
    plt.xlabel("Feature Number")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_y1 = max(data[0])
    max_y2 = max(data[1])
    max_y3 = max(data[2])
    max_y4 = max(data[3])
    max_x1 = data[0].index(max(data[0]))
    max_x2 = data[1].index(max(data[1]))
    max_x3 = data[2].index(max(data[2]))
    max_x4 = data[3].index(max(data[3]))
    plt.text(max_x1, max_y1, str(max_x1 + 1) + '(' + str(max_y1 * 100) + '%)', fontsize=10, c='#1f77b4')
    plt.text(max_x2, max_y2, str(max_x2 + 1) + '(' + str(round(max_y2 * 100, 1)) + '%)', fontsize=10, c='#ff7f0e')
    plt.text(max_x3, max_y3, str(max_x3 + 1) + '(' + str(max_y3 * 100) + '%)', fontsize=10, c='#2ca02c')
    plt.text(max_x4, max_y4, str(max_x4 + 1) + '(' + str(max_y4 * 100) + '%)', fontsize=10, c='#d62728')
    plt.legend(loc="lower right")
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()

def plot_bar(pc1_featurescore, out, number=10):
    # 中文乱码和坐标轴负号处理
    matplotlib.rc('font', family='SimHei', weight='bold')
    plt.rcParams['axes.unicode_minus'] = False
    feature = (list(pc1_featurescore['Feature']))
    data = list(pc1_featurescore['PC1_loading_abs'])
    if len(pc1_featurescore) > 10 and number != False:
        feature = (list(pc1_featurescore['Feature'])[:number-1] + ['其他特征'])
        data = list(pc1_featurescore['PC1_loading_abs'])[:number-1] + [sum(list(pc1_featurescore['PC1_loading_abs'])[9:])]
    feature.reverse()
    data.reverse()
    # 绘图
    fig, ax = plt.subplots()
    ax.barh(range(len(feature)), data)
    #设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(feature)))
    ax.set_yticklabels(feature)
    #不要X横坐标上的label标签。
    plt.xticks(())
    plt.title('Feature Weight')
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()

def plot_turf_bar(pc1_featurescore, out, number=10):
    # 中文乱码和坐标轴负号处理
    matplotlib.rc('font', family='SimHei', weight='bold')
    plt.rcParams['axes.unicode_minus'] = False
    feature = (list(pc1_featurescore['Feature']))
    data = list(pc1_featurescore['Weight'])
    if len(pc1_featurescore) > 10 and number != False:
        feature = (list(pc1_featurescore['Feature'])[:number-1] + ['其他特征'])
        data = list(pc1_featurescore['Weight'])[:number-1] + [sum(list(pc1_featurescore['Weight'])[9:])]
    feature.reverse()
    data.reverse()
    # 绘图
    fig, ax = plt.subplots()
    ax.barh(range(len(feature)), data)
    #设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(feature)))
    ax.set_yticklabels(feature)
    #不要X横坐标上的label标签。
    plt.xticks(())
    plt.title('Feature Weight')
    # 保存图片
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(out+".tiff")
    png1.close()
