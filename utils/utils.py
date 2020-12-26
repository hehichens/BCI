"""
some useful tools
edit by hichens
"""

from scipy import signal
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings; warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from xgboost import XGBClassifier


## 段波功率
def bandpowers(segment):
    features = []
    for i in range(len(segment)):
        f,Psd = signal.welch(segment[i,:], 100)
        power1 = 0
        power2 = 0
        f1 = []
        for j in range(0,len(f)):
            if(f[j]>=4 and f[j]<=13):
                power1 += Psd[j]
            if(f[j]>=14 and f[j]<=30):
                power2 += Psd[j]
        features.append(power1)
        features.append(power2)
    return features


## 离散余弦变换
from scipy.fftpack import fft, dct
def dct_features(segment):
    features = []
    for i in range(len(segment)):
        dct_coef = dct(segment[i,:], 2, norm='ortho')
        power = sum( j*j for j in dct_coef)
        features.append(power)
    return features


##小波特征
def wavelet_features(epoch):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    features = []
    for i in range(len(epoch)):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)		#calculating the coefficients of wavelet transform.
    for x in range(len(epoch)):   
        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        features.append(abs(np.sum(np.square(cA_values[x]))))
        
    for x in range(len(epoch)):      
        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        features.append(abs(np.sum(np.square(cD_values[x]))))
        
    return features


def csp_features():
    pass


## test in different models
def test_model(df):
    """
    df: DataFrame format data
    """
    features, labels = df.iloc[:, :-1].values, df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, \
        shuffle=True, test_size=0.3, random_state=42)

    ## classify model
    clfs = [
        # Bayes Method
        GaussianNB(priors=None, var_smoothing=1e-9),
        MultinomialNB(alpha=0.1, fit_prior=True, class_prior=None),
        BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
        KNeighborsClassifier(n_neighbors=1, 
                            weights='uniform',  # uniform、distance
                            algorithm='auto',  # {‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}
                            leaf_size=10, 
                            # p=1, 
                            metric='minkowski', 
                            metric_params=None, 
                            n_jobs=None),
        
        SVC(C=1e7, kernel='rbf',
                    shrinking=True, 
                    probability=False, 
                    tol=1e-2,
                    cache_size=200, 
                    class_weight=None, 
                    verbose=False, 
                    # max_iter=1e5, 
                    # decision_function_shape='ovr', 
                    random_state=42),
                    
        GaussianProcessClassifier(kernel=1.0 * RBF(64.0), warm_start=True,
                                    n_restarts_optimizer=0, 
                                    max_iter_predict=100),

        DecisionTreeClassifier(criterion='entropy',  # entropy, gini, 
                            splitter='random',  # best, random
                            max_depth=16, 
                            min_samples_split=2,
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=42, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            class_weight=None),
        
        RandomForestClassifier(n_estimators=256, 
                                criterion='entropy', 
                                max_depth=None, 
                                min_samples_split=2, 
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                bootstrap=True, 
                                oob_score=False,
                                n_jobs=1, 
                                verbose=0, 
                                warm_start=False, 
                                class_weight=None),
        
        AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=128, criterion='entropy', max_depth=None), 
                            n_estimators=32, 
                            learning_rate=1e-2, 
                            algorithm='SAMME.R', # SAMME.R, SAMME.R
                            random_state=42),
        
        LinearDiscriminantAnalysis(solver='svd', # {‘svd’, ‘lsqr’, ‘eigen’}
                                    priors=None,
                                    n_components=None,
                                    tol=1e-4),

        QuadraticDiscriminantAnalysis(priors=None, 
                                        reg_param=0.0, 
                                        store_covariance=False, 
                                        tol=1e-4),
        
        XGBClassifier(learning_rate=None,
                        booster='gbtree', # gbtree, gblinear, dart
                        n_estimators=2000,           # 树的个数
                        # max_depth=10,               # 树的深度 
                    #   min_child_weight = 1,      # 叶子节点最小权重
                    #   gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                    #   subsample=1,               # 所有样本建立决策树
                    #   colsample_btree=1,         # 所有特征建立决策树
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                    #   slient = 0,
                        # reg_alpha=1e-4,
                        # reg_lambda=1e-4,
                        # use_label_encoder=False,
                        random_state=42,
                        n_jobs=8
        ),

        MLPClassifier(hidden_layer_sizes=128, 
                    activation='relu',  # {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                    solver='adam',  # {‘lbfgs’, ‘sgd’, ‘adam’}
                    alpha=1e-3, 
                    batch_size=256,  # 
                    learning_rate='invscaling', # {‘constant’, ‘invscaling’, ‘adaptive’}
                    learning_rate_init=1e-3, 
                    power_t=0.5, 
                    max_iter=10000, 
                    shuffle=True, 
                    random_state=42, 
                    tol=1e-7, 
                    verbose=False, 
                    warm_start=False, 
                    momentum=0.9, 
                    nesterovs_momentum=True, 
                    early_stopping=False, 
                    validation_fraction=0.1, 
                    beta_1=0.9, 
                    beta_2=0.999, 
                    epsilon=1e-08, 
                    n_iter_no_change=10, 
                    max_fun=15000)
    ]
    
    ## train and print(accuracy)
    print("="*40, "result", "="*40)
    print("%40s %10s %10s"%("Model        ", "Accuracy", "time"))
    result = []
    weights = [] # model vote weight
    score_list = []
    train_score_list = []
    for clf in clfs:
        start = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_pred, y_test)
        train_score = clf.score(X_train, y_train)
        score_list.append(score)
        train_score_list.append(train_score)
        weights.append(score)
        print("%40s %5.4f|%5.4f %10.2f s"%(type(clf).__name__, train_score, score, time.time() - start))
        result.append([type(clf).__name__, score])
    print("%40s %5.4f|%5.4f %10.2f s"%("Average", np.mean(train_score_list), np.mean(score_list), 0.0))
    result.append(["Average", np.mean(score)])

    ## model merge
    N = len(weights)
    split_index = sum(weights) / 2
    pred = np.zeros(len(y_pred))
    start = time.time()
    for i, clf in enumerate(clfs):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pred += weights[i] * y_pred
    
    pred[pred<=split_index] = 0
    pred[pred>split_index] = 1
    score = accuracy_score(pred, y_test)
    print("%40s %10.4f %10.2f s"%("Model Stack", score, time.time() - start))
    result.append(["Model Stack", score])
    
    print("="*40, "end", "="*40)
    return result

    
def save_csv(result, path=None):
    res_df = pd.DataFrame(result)
    res_df.columns = ['method', 'score']
    res_df.to_csv(path, index=None)


## plot and show
def plot(data):
    pass