from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from joblib import load as jobld
from joblib import dump as jobdp
from math import sqrt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def ROC(X_train, y_train):
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
    plt.show()
    return mean_auc

def Performance(y_t, y_p):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    y_t = list(y_t)
    y_p = list(y_p)
    for i in range(len(y_t)):
        t_label = y_t[i]
        p_label = y_p[i]
        if t_label == p_label == 0:
            tp += 1
        if t_label == p_label == 1:
            tn += 1
        if t_label == 0 and p_label == 1:
            fn += 1
        if t_label == 1 and p_label == 0:
            fp += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    ppv = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
    return [tp, tn, fp, fn, acc, sn, sp, ppv, mcc]

def PRF(tp,tn,fp,fn):
    pre = tp / (tp + fp)
    rcl = tp / (tp + fn)
    f1s = (2 * pre * rcl) / (pre + rcl)
    return [round(pre, 5), round(rcl, 5), round(f1s, 5)]

def V_to_L(value, lab1=0, lab2=1, threshold=0.5):
    out = []
    for i in value:
        if i[0] >= 0.5:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)

def svm_grid(train_data, train_label):
    my_svm = svm.SVC(decision_function_shape="ovo", random_state=0)
    c_number = []
    for i in range(-5, 15 + 1, 2):
        c_number.append(2 ** i)
    gamma = []
    for i in range(-15, 3 + 1, 2):
        gamma.append(2 ** i)
    parameters = {'C': c_number, 'gamma': gamma}
    new_svm = GridSearchCV(my_svm, parameters, cv=5, scoring="accuracy", return_train_score=False, n_jobs=1)
    model = new_svm.fit(train_data, train_label)
    best_c = model.best_params_['C']
    best_g = model.best_params_['gamma']
    return best_c, best_g

def MySVM(X_train, y_train, tx=pd.DataFrame(), ty=pd.DataFrame(), model=None, out=None, c=None, g=None, cv=None, threshold=0.5):
    if model == None:
        if c != None and g != None:
            c, g = c, g
        else:
            c, g = svm_grid(X_train, y_train)
        model = svm.SVC(kernel='rbf', C=c, gamma=g, probability=True)
        if cv != None:
            predict_label = cross_val_predict(model, X_train, y_train, cv=cv)
            if out == None:
                return Performance(y_train, predict_label)
            else:
                return predict_label
        model.fit(X_train, y_train)
        if out != None:
            jobdp(model, out)
        if tx.empty == True and ty.empty == True:
            return model
        else:
            predict_value = model.predict_proba(tx)
            predict_label = V_to_L(predict_value, threshold)
            return Performance(ty, predict_label)
    else:
        if type(model) == str:
            model = jobld(model)
        predict_value = model.predict_proba(X_train)
        if out == None:
            predict_label = V_to_L(predict_value)
            return Performance(y_train, predict_label)
        else:
            return predict_value

def MyKNN(X_train, y_train, tx=pd.DataFrame(), ty=pd.DataFrame(), model=None, out=None, cv=None, threshold=0.5):
    if model == None:
        model = KNeighborsClassifier(n_neighbors=14,p=1)
        if cv != None:
            predict_label = cross_val_predict(model, X_train, y_train, cv=cv)
            if out == None:
                return Performance(y_train, predict_label)
            else:
                return predict_label
        model.fit(X_train, y_train)
        if out != None:
            jobdp(model, out)
        if tx.empty == True and ty.empty == True:
            return model
        else:
            predict_value = model.predict_proba(tx)
            predict_label = V_to_L(predict_value, threshold)
            return Performance(ty, predict_label)
    else:
        if type(model) == str:
            model = jobld(model)
        predict_value = model.predict_proba(X_train)
        if out == None:
            predict_label = V_to_L(predict_value)
            return Performance(y_train, predict_label)
        else:
            return predict_value
