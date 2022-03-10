import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

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
