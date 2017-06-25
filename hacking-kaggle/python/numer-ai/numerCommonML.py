from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import warnings
from functools import wraps

import pandas as pd
import psutil
import scipy.io
import scipy.io.wavfile
import sklearn.linear_model as slm
import xgboost as xgb
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.preprocessing import Normalizer
# from tsne import bh_sne
from xgboost import XGBClassifier



warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import scipy

import os,sys

from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
import pandas
from scipy.stats import norm
import scipy.sparse
# from bhtsne import tsne
import spectrum

from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import PolynomialFeatures, Imputer
from sklearn.neighbors import KNeighborsClassifier

import random
import numpy as np


import datetime
import time

EVAL_TYPE = 'neg_log_loss'
TARGET_VAR= 'target'
POLY_DEGREE=2
BINARYLOGISTIC = 'binary:logistic'

TOURNAMENT_DATA_CSV = '/numerai_tournament_data.csv'
TRAINING_DATA_CSV = '/numerai_training_data.csv'
TEST_FEATURES_CSV = '/numerai_tournament_data-features.csv'
TRAIN_FEATURES_CSV = '/numerai_training_data-features.csv'

BASE_FOLDER = '/home/oyama/db/Dropbox/dev2/books/2019/BML/kaggle/numer-new/data/'

class NumerCommonML(object):

    ############################ MODELS #####################################################
    C = 0.005
    lr_best_params = {'penalty': 'l2', 'C': C, 'solver': 'newton-cg', 'fit_intercept': False}
    M_LR_OPT = LogisticRegression(**lr_best_params)
    M_LR = LogisticRegression()
    M_SGD=SGDClassifier(loss = 'modified_huber',penalty = 'elasticnet',alpha = 1e-3,n_iter = 10)
    M_SGD2= SGDClassifier(loss="log", penalty="elasticnet")
    M_SGD3 = SGDClassifier(loss='log', penalty='elasticnet', alpha=5e-1, n_iter=10)
    M_TSVD = TruncatedSVD(n_components=40, random_state=0)
    M_PCA_OPT = PCA(copy=True, iterated_power='auto', n_components=None, random_state=None, svd_solver='auto', tol=0.0, whiten=True)
    M_PCA = PCA()
    M_RF=RandomForestClassifier(n_estimators=250)
    M_RIDGE = slm.Ridge()
    M_POLY=PolynomialFeatures(degree=POLY_DEGREE,interaction_only=False)
    M_SCALER=MinMaxScaler()
    # M_SCALER=Normalizer( norm = 'max' )
    M_SCALER_MAX = Normalizer(norm='max')
    # M_ADABOOST = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=3), n_estimators=10)

    mx_depth = 50
    M_XG= xgb.XGBClassifier(base_score=0.5, colsample_bytree=0.5,
                            gamma=0.017, learning_rate=0.15, max_delta_step=0,
                            max_depth=mx_depth, min_child_weight=3, n_estimators=300,
                            nthread=-1, objective=BINARYLOGISTIC, seed=0,
                            silent=1, subsample=0.9)
    M_KNN=KNeighborsClassifier()



    ############################ MODELS #####################################################


    ############################ PIPELINES #####################################################

    from sklearn.calibration import CalibratedClassifierCV
    # in our case, 'isotonic' works better than default 'sigmoid'

    PIP_BEST= make_pipeline(
        M_POLY,
        M_SCALER,
        M_SGD,
        M_LR_OPT
    )


    PIP_BEST2 = make_pipeline(
        M_POLY,
        M_SCALER,
        M_SGD2,
        M_LR_OPT
    )

    PIP_BEST3 = make_pipeline(
        M_POLY,
        M_SCALER,
        M_SGD3,
        M_LR_OPT
    )

    PIP_BEST4 = make_pipeline(
        M_POLY,
        M_SCALER,
        M_TSVD,
        M_SGD,
        M_SGD2,
        M_LR_OPT,
        # M_XG,
    )

    # knn_params = dict(
    #     clf__n_neighbors=[1, 3, 5, 7, 15],
    #     clf__weights=['distance', 'uniform'],
    # )
    # clf = Pipeline([
    #     ('vec', M_SCALER),
    #     ('clf', M_KNN),
    #     ('lr', M_LR)
    # ])
    # PIP_KNN = GridSearchCV(clf, knn_params, scoring='roc_auc', verbose=True, cv=5)


    PIP_CALIBRATED_LR_ISOTONIC_1 = CalibratedClassifierCV(PIP_BEST, method='isotonic', cv=5)
    PIP_CALIBRATED_LR_ISOTONIC_2 = CalibratedClassifierCV(PIP_BEST2, method='isotonic', cv=5)
    PIP_CALIBRATED_LR_ISOTONIC_3 = CalibratedClassifierCV(PIP_BEST3, method='isotonic', cv=5)

    PIP_CALIBRATED_LR_SIGMOID_1 = CalibratedClassifierCV(PIP_BEST, method='sigmoid', cv=5)
    PIP_CALIBRATED_LR_SIGMOID_2 = CalibratedClassifierCV(PIP_BEST2, method='sigmoid', cv=5)
    PIP_CALIBRATED_LR_SIGMOID_3 = CalibratedClassifierCV(PIP_BEST3, method='sigmoid', cv=5)


    PIP_BASE = Pipeline(steps=[
        # ('poly', M_POLY),
        ('pca', M_PCA),
        ('scaler', M_SCALER),
    ])

    PIP_LR = Pipeline(steps=[
        ('base', PIP_BASE),
        ('lr', M_LR)
    ])

    PIP_LR_MAX_SCALER = Pipeline(steps=[
        ('pca', M_PCA_OPT),
        ('MAXSCALE', M_SCALER_MAX),
        ('lr', M_LR)
    ])

    plr = make_pipeline(M_PCA, M_SCALER, M_LR)
    PIP_GRID_LR = GridSearchCV(plr, dict(logisticregression__penalty=['l2'],
                                         logisticregression__C=[0.00001, 0.001, 0.1, 1, 2, 5],
                                         logisticregression__solver=['newton-cg'],
                                         logisticregression__fit_intercept=[False, True]
                                         ),
                               scoring=EVAL_TYPE, verbose=True, refit=True, cv=5)

    plr2 = make_pipeline(M_LR)
    PIP_GRID_LR_PURE = GridSearchCV(plr2, dict(
        logisticregression__penalty=['l2'],
        logisticregression__C=[0.0001, 0.001, 0.1,1, 2],
        logisticregression__solver=['newton-cg'],
        logisticregression__fit_intercept=[False, True]
        # logisticregression__loss=['neg_log_loss']
    ),
                                scoring=EVAL_TYPE, verbose=True, refit=True, cv=5)

    ############################ PIPELINES #####################################################

    # @fn_timer
    @staticmethod
    def loadFeaturesSplit(r_state, split=True):
        df_train = pd.read_csv(BASE_FOLDER + TRAIN_FEATURES_CSV)
        df_test = pd.read_csv(BASE_FOLDER + TEST_FEATURES_CSV)

        if split == False:
            return df_train, df_test

        feature_cols = list(df_train.columns[:-1])
        target_col = df_train.columns[-1]

        X, y = df_train[feature_cols], df_train[target_col]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=r_state)

        X_test = df_test[feature_cols].values
        return X_test, X_train, X_valid, df_test, y_train, y_valid

    # @fn_timer
    @staticmethod
    def loadDataSplit(r_state, split=True):
        df_train = pd.read_csv(BASE_FOLDER + TRAINING_DATA_CSV)
        df_test = pd.read_csv(BASE_FOLDER + TOURNAMENT_DATA_CSV)

        answers_1_SINGLE = df_train[TARGET_VAR]
        df_train = df_train.drop(TARGET_VAR, axis=1)
        df_train = df_train.drop('id', axis=1)
        df_train = df_train.drop('era', axis=1)
        df_train = df_train.drop('data_type', axis=1)
        df_train = pd.concat([df_train, answers_1_SINGLE], axis=1)
        print(df_train.head(1))

        tid_1_SINGLE = df_test['id']
        df_test = df_test.drop('id', axis=1)
        df_test = df_test.drop('era', axis=1)
        df_test = df_test.drop('data_type', axis=1)
        df_test = pd.concat([tid_1_SINGLE, df_test], axis=1)

        if split==False:
            return df_train, df_test

        feature_cols = list(df_train.columns[:-1])
        target_col = df_train.columns[-1]

        X, y = df_train[feature_cols], df_train[target_col]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=r_state)

        X_test = df_test[feature_cols].values
        return X_test, X_train, X_valid, df_test, y_train, y_valid

    @staticmethod
    def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

    @staticmethod
    def createNewDir(BASE_FOLDER):
        parquet_dir = os.path.join(BASE_FOLDER, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(parquet_dir)
        return parquet_dir

    @staticmethod
    def gridBestParams(train_X, train_y, modelType, folds=5):
        print('running grid:' + str(modelType))

        params_lr = {'penalty': ['l2'], 'C': [0.0001, 0.001, 0.1, 0.01, 1, 2, 5],
                     'solver': ['newton-cg'],
                     'fit_intercept': [False, True], 'loss': ['neg_log_loss']}

        params_enet = {
            'loss': ['log'],
            'penalty': ['elasticnet'],
            'n_iter': [5],
            'alpha': np.logspace(-4, 4, 10),
            'l1_ratio': [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15, 0.2]
        }

        model_lg = LogisticRegression()
        model_enet = SGDClassifier()
        if modelType == 'lr':
            method = model_lg
            params = params_lr
        else:
            method = model_enet
            params = params_enet

        print('running grid:' + str(params))

        gscv = GridSearchCV(method, params, scoring='neg_log_loss', cv=folds, verbose=1)
        gscv.fit(train_X, train_y)
        for params, mean_score, all_scores in gscv.grid_scores_:
            print('{:.6f} (+/- {:.6f}) for {}'.format(mean_score, all_scores.std() / 2, params))
        print('params:{params}'.format(params=gscv.best_params_))
        print('score:{params}'.format(params=gscv.best_score_))
        return gscv.best_params_

    @staticmethod
    def savePred(df_pred, name, loss):
        csv_path = BASE_FOLDER + '/pred/p_{}_{}_{}.csv'.format(loss, name, (str(time.time())))
        df_pred.to_csv(csv_path, columns=('id', 'probability'), index=None)
        # print('Saved: {}'.format(csv_path))
        # print('... check data:', df_pred.ix[:5, :])

    @staticmethod
    def featureFFT(y):
        fs = 400
        n = len(y)
        dt = 1 / float(fs)  # Get time resolution

        fft_output = np.fft.rfft(y)  # Perform real fft
        rfreqs = np.fft.rfftfreq(n, dt)  # Calculatle frequency bins
        fft_mag = np.abs(fft_output)  # Take only the magnitude of the spectrum
        fft_mag = fft_mag * 2 / n

        return fft_mag

    @staticmethod
    def featureAR(ch):
        ar_coeffs, dnr, reflection_coeffs = spectrum.aryule(ch, order=8)
        return np.abs(ar_coeffs)

    @staticmethod
    def featureEnt( row, base=2):
        x_ent= -((np.log(row) / np.log(base)) * row).sum(axis=0)
        return x_ent

    @staticmethod
    def featureShannonEnt(row):
        row=row.div(row.sum())
        # print (row)
        return -sum([p * math.log(p) for p in row if p != 0])

    @staticmethod
    def enrichFeatures( row):
        # print (len(row))
        x_fft = NumerCommonML.featureFFT(row)
        x_ent = NumerCommonML.featureShannonEnt(row)
        x_ar = NumerCommonML.featureAR(row)
        s = pd.Series({'x_ent': x_ent, 'ar1': x_ar[0], 'ar2': x_ar[1], 'ar3': x_ar[2], 'ar4': x_ar[3], 'ar5': x_ar[4],
                       'ar6': x_ar[5], 'ar7': x_ar[6], 'ar8': x_ar[7],
                       'x_fft1': x_fft[0], 'x_fft2': x_fft[1], 'x_fft3': x_fft[2], 'x_fft4': x_fft[3],
                       'x_fft5': x_fft[4], 'x_fft6': x_fft[5], 'x_fft7': x_fft[6], 'x_fft8': x_fft[7],
                       'x_fft9': x_fft[8], 'x_fft10': x_fft[9],'x_fft11': x_fft[10]})
        # print (s)
        return s

    @staticmethod
    def genBasicFeatures(inDF):
        print('Generating features ...')
        df_copy=inDF.copy(deep=True)
        magicNumber=21
        feature_cols = list(inDF.columns[:-1])
        # feature_cols=xgb_cols
        target_col = inDF.columns[-1]

        inDF['x_mean'] = np.mean(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_median'] = np.median(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_std'] = np.std(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_skew'] = scipy.stats.skew(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_kurt'] = scipy.stats.kurtosis(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_var'] = np.var(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_max'] = np.max(df_copy.ix[:, 0:magicNumber], axis=1)
        inDF['x_min'] = np.min(df_copy.ix[:, 0:magicNumber], axis=1)
        # http://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
        inDF=inDF.merge(df_copy.ix[:, 0:magicNumber].apply(lambda row: NumerCommonML.enrichFeatures(row), axis=1),
                        left_index=True, right_index=True)


        print (inDF.head(1))
        return inDF

    # @fn_timer
    @staticmethod
    def makeFeatures(BASE_FOLDER):
        NumerCommonML.cpuStats()
        print ('makeFeatures::Base folder:' + BASE_FOLDER)
        df_train = pd.read_csv(BASE_FOLDER + TRAINING_DATA_CSV)
        df_test = pd.read_csv(BASE_FOLDER + TOURNAMENT_DATA_CSV)

        answers_1_SINGLE = df_train[TARGET_VAR]
        df_train = df_train.drop(TARGET_VAR, axis=1)
        df_train = df_train.drop('id', axis=1)
        df_train = df_train.drop('era', axis=1)
        df_train = df_train.drop('data_type', axis=1)
        df_train = NumerCommonML.genBasicFeatures(df_train)
        df_train=pd.concat([df_train, answers_1_SINGLE], axis=1)
        print(df_train.head(1))


        tid_1_SINGLE = df_test['id']
        df_test = df_test.drop('id', axis=1)
        df_test = df_test.drop('era', axis=1)
        df_test = df_test.drop('data_type', axis=1)
        df_test = NumerCommonML.genBasicFeatures(df_test)
        df_test = pd.concat([tid_1_SINGLE, df_test], axis=1)


        df_train.to_csv(BASE_FOLDER + TRAIN_FEATURES_CSV, index_label=False, index=False)
        df_test.to_csv(BASE_FOLDER + TEST_FEATURES_CSV, index_label=False, index=False)
        print('Done makeFeatures.')
        # NumerCommon.cpuStats()

if __name__ == '__main__':

    NumerCommonML.makeFeatures(BASE_FOLDER)

