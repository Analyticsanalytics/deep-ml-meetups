import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob
import re
import copy
import pywFM
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, RandomizedLogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, log_loss)
from sklearn.cross_validation import train_test_split

from sklearn import preprocessing as pre
from sklearn.preprocessing import Imputer
from sklearn import linear_model, tree, lda

# from tpot import TPOTClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import pylab as pl
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import svm
random.seed(2016)
np.random.seed(2016)
from skbayes import linear_models
from skbayes.linear_models.bayes_logistic import *

from sklearn.utils.estimator_checks import check_estimator

from IeegConsts import *
from IeegConsts import *
from IeegFeatures import *

from MVC import *

# from mlxtend import classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

print DATA_FOLDER


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def intersect(a, b):
    return list(set(a) & set(b))


from sys import platform


def mat_to_pandas(path):
    if platform == "linux" or platform == "linux2":
        mat = loadmat(path, verify_compressed_data_integrity=False)
    elif platform == "darwin":
        mat = loadmat(path)

    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    sequence = -1
    if 'sequence' in names:
        sequence = mat['dataStruct']['sequence']
    # return pd.DataFrame(remove_dc(ndata['data']), columns=ndata['channelIndices'][0]), sequence
    return pd.DataFrame((ndata['data']), columns=ndata['channelIndices'][0]), sequence


def get_best_estimator_by_grid_search(train_X, train_y, modelType):
    params_lr = {'penalty': ['l2'], 'C': [1, 2, 3, 4, 5, 10, 20, 50, 80, 100, 200, 500, 5000],
                 'solver': ['newton-cg'],
                 'fit_intercept': [False, True]}
    model_lg = LogisticRegression()

    if modelType == 'lr':
        method = model_lg
        params = params_lr
    print 'running grid:' + str(params)

    gscv = grid_search.GridSearchCV(method, params, scoring='roc_auc', cv=4)
    gscv.fit(train_X, train_y)
    for params, mean_score, all_scores in gscv.grid_scores_:
        print('{:.6f} (+/- {:.6f}) for {}'.format(mean_score, all_scores.std() / 2, params))
    print('params:{params}'.format(params=gscv.best_params_))
    print('score:{params}'.format(params=gscv.best_score_))
    return gscv.best_params_


def run_kfold(nfolds, train, test, features, target, random_state=87 * 14 * 36 * 19):
    yfull_train = dict()
    yfull_test = copy.deepcopy(test[['id']].astype(object))

    unique_sequences = np.array(train['sequence_id'].unique())
    kf = KFold(len(unique_sequences), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    for train_seq_index, test_seq_index in kf:
        num_fold += 1
        print('Start fold {} from {}'.format(num_fold, nfolds))
        train_seq = unique_sequences[train_seq_index]
        valid_seq = unique_sequences[test_seq_index]
        print('Length of train people: {}'.format(len(train_seq)))
        print('Length of valid people: {}'.format(len(valid_seq)))

        X_train, X_valid = train[train['sequence_id'].isin(train_seq)][features], \
                           train[train['sequence_id'].isin(valid_seq)][features]
        y_train, y_valid = train[train['sequence_id'].isin(train_seq)][target], \
                           train[train['sequence_id'].isin(valid_seq)][target]

        X_test = test[features]

        X_test = X_test.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
        X_train = X_train.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
        y_train = y_train.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
        X_valid = X_valid.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
        y_valid = y_valid.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

        print('Length train:', len(X_train))
        print('Length valid:', len(X_valid))

        lr_best_params = {'penalty': 'l2', 'C': 10, 'solver': 'newton-cg', 'fit_intercept': True}
        # # lr_best_params = get_best_estimator_by_grid_search(X_train, y_train, modelType='lr')
        lr = LogisticRegression(**lr_best_params)
        # # lr = LogisticRegression()
        # blr = EBLogisticRegression()
        # ld = lda.LDA()
        # # VBLogisticRegression

        mx_depth = 3000
        xg = xgb.XGBClassifier(base_score=0.5, colsample_bytree=0.5,
                               gamma=0.017, learning_rate=0.015, max_delta_step=0,
                               max_depth=mx_depth, min_child_weight=3, n_estimators=20000,
                               nthread=-1, objective='binary:logistic', seed=0,
                               silent=1, subsample=0.9)

        # clfs = [xg, xg1,xg2,xg3]
        # clf_labels = ['lr','xg','ld','blr']
        #
        # eclf_B = MajorityVoteClassifier(classifiers=clfs, weights=[1,1,1,1])
        # tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)

        # clf =svm.SVC(gamma=0.001, C=100., probability=True)
        clf =lr
        clf.fit(X_train, y_train)
        yhat = clf.predict_proba(X_valid)

        for i in range(len(X_valid.index)):
            yfull_train[X_valid.index[i]] = yhat[i]

        print("Validating...")
        check = clf.predict_proba(X_valid)
        score = roc_auc_score(y_valid.tolist(), check[:, 1])
        print('*** VALIDATION ROC_AUC: {:.6f}'.format(score))
        print("Predict test set...")
        test_prediction = clf.predict_proba(X_test)
        yfull_test['kfold_' + str(num_fold)] = test_prediction[:, 1]

    # Copy dict to list
    train_res = []
    for i in range(len(train.index)):
        train_res.append(yfull_train[i])

    score = roc_auc_score(train[target], np.array(train_res)[:, 1])
    print('Final AUC: {:.8f}'.format(score))

    # Find mean for KFolds on test
    merge = []
    for i in range(1, nfolds + 1):
        merge.append('kfold_' + str(i))
    yfull_test['mean'] = yfull_test[merge].mean(axis=1)
    return yfull_test['mean'].values, score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    total = 0
    for id in test['id']:
        patient = id // 100000
        fid = id % 100000
        str1 = "new_" + str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('id')
    # output.remove('file_size')
    return sorted(output)


def bestFeaturesPCA(last_cols, maxFeatures=40):
    # rlasso = RandomizedLasso(alpha=0.025)
    # algo_=rlasso
    F_NAME_TRAIN = TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL + '-feat_TRAIN_df.csv'
    X_df_train = pandas.read_csv(F_NAME_TRAIN, engine='python')
    X_df_train_SINGLE = X_df_train.copy(deep=True)
    X_df_train_SINGLE = X_df_train_SINGLE.sort(['patient_id'], ascending=[True])
    X_df_train_SINGLE.drop('id', axis=1, inplace=True)
    X_df_train_SINGLE = dropBadFiles(X_df_train_SINGLE)
    # print 'Final shape 222******:' + str(X_df_train.shape)
    X_df_train_SINGLE.drop('file', axis=1, inplace=True)
    X_df_train_SINGLE.drop('patient_id', axis=1, inplace=True)
    print X_df_train_SINGLE.shape
    # X_df_train_SINGLE = X_df_train_SINGLE.loc[X_df_train_SINGLE['file_size'] > 1500000]
    answers_1_SINGLE = list(X_df_train_SINGLE[singleResponseVariable].values)
    answers_1_SINGLE = map(int, answers_1_SINGLE)
    X_df_train_SINGLE = X_df_train_SINGLE.drop(singleResponseVariable, axis=1)
    print X_df_train_SINGLE.shape
    X_df_train_SINGLE.drop('file_size', axis=1, inplace=True)
    X_df_train_SINGLE.drop('sequence_id', axis=1, inplace=True)
    X_df_train_SINGLE.drop('segment', axis=1, inplace=True)

    X_df_train_SINGLE = X_df_train_SINGLE[last_cols]
    ch2 = PCA(n_components=maxFeatures)
    ch2 = SelectKBest(k=maxFeatures)
    X_train_features = ch2.fit_transform(X_df_train_SINGLE, answers_1_SINGLE)
    # X_train_features = X_df_train_SINGLE[:, b.get_support()]

    # pca = PCA(n_components=maxFeatures, whiten=True)
    # pca.fit(X_df_train_SINGLE)
    # train_data = pca.transform(X_df_train_SINGLE)


    fff = np.asarray(X_df_train_SINGLE.columns)[ch2.get_support()]
    # normalize data
    # df_norm = (X_df_train_SINGLE - X_df_train_SINGLE.mean()) / X_df_train_SINGLE.std()
    # svd = TruncatedSVD(n_components=50, random_state=42)


    return fff
    # top_ranked_features = sorted(enumerate(ch2.scores_), key=lambda x: x[1], reverse=True)[:maxFeatures]

    # f_union = FeatureUnion([("pca", pca), ("kbest", kbest)])
    # f_union = FeatureUnion([("pca", kbest)])
    # pipeline = Pipeline([('f_union', f_union)])
    # pipeline.fit(X_df_train_SINGLE, answers_1_SINGLE)
    # support = pipeline.named_steps['f_union']

    # feature_names = np.array(last_cols)
    # return  feature_names[support]

    # selected_feat = f_union.fit(X_df_train_SINGLE, answers_1_SINGLE).transform(X_df_train_SINGLE)

    # return top_ranked_features


def bestFatures(last_cols, maxFeatures):
    mx_depth = 100
    algo_xgbm = xgb.XGBClassifier(base_score=0.5, colsample_bytree=0.5,
                                  gamma=0.017, learning_rate=0.15, max_delta_step=0,
                                  max_depth=mx_depth, min_child_weight=3, n_estimators=2000,
                                  nthread=-1, objective='binary:logistic', seed=0,
                                  silent=1, subsample=0.9)

    algo_ = algo_xgbm
    X_df_train = pd.read_hdf(TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL
                             + 'X_df_train.hdf', 'data', format='fixed', complib='blosc', complevel=9)

    X_df_train_SINGLE = X_df_train.copy(deep=True)
    X_df_train_SINGLE = dropBadFiles(X_df_train_SINGLE)

    # X_df_train_SINGLE=X_df_train_SINGLE.sort(['patient_id','segment'], ascending=[True, True])
    X_df_train_SINGLE = X_df_train_SINGLE.sort(['patient_id'], ascending=[True])

    X_df_train_SINGLE.drop('id', axis=1, inplace=True)
    X_df_train_SINGLE.drop('file', axis=1, inplace=True)
    X_df_train_SINGLE.drop('patient_id', axis=1, inplace=True)
    print X_df_train_SINGLE.shape
    # X_df_train_SINGLE = X_df_train_SINGLE.loc[X_df_train_SINGLE['file_size'] > 500000]
    answers_1_SINGLE = list(X_df_train_SINGLE[singleResponseVariable].values)
    answers_1_SINGLE = map(int, answers_1_SINGLE)
    X_df_train_SINGLE = X_df_train_SINGLE.drop(singleResponseVariable, axis=1)

    print X_df_train_SINGLE.shape
    X_df_train_SINGLE.drop('file_size', axis=1, inplace=True)
    X_df_train_SINGLE.drop('sequence_id', axis=1, inplace=True)
    X_df_train_SINGLE.drop('segment', axis=1, inplace=True)

    X_df_train_SINGLE = X_df_train_SINGLE[last_cols]

    X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

    trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=.11)  # CV
    model_train = algo_.fit(trainX, trainY, early_stopping_rounds=100, eval_metric="auc", eval_set=[(testX, testY)],
                            verbose=False)
    # print model_train
    predictions = algo_.predict_proba(testX)[:, 1]

    print 'ROC AUC:' + str(roc_auc_score(testY, predictions))
    print 'LOG LOSS:' + str(log_loss(testY, predictions))

    # Build pasty expression -- feed best features automatically
    glm_factor = pd.Series(algo_.booster().get_fscore()).sort_values(ascending=False)
    glm_factor = glm_factor.head(maxFeatures)
    glm_factor = list(glm_factor.index)
    return glm_factor


def read_test_train():
    print("Load train.csv...")

    train_dir = TRAIN_DATA_FOLDER_IN_ALL
    test_dir = TEST_DATA_FOLDER_IN_ALL

    # Train
    ieegFeaturesTrain = IeegFeatures(train_dir, True)
    df_cols_train = ieegFeaturesTrain.ieegGenCols()
    print len(df_cols_train)
    F_NAME_TRAIN = TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL + '-feat_TRAIN_df.csv'
    X_df_train = pandas.read_csv(F_NAME_TRAIN, engine='python')
    X_df_train.drop('Unnamed: 0', axis=1, inplace=True)
    # answers_1_SINGLE = list(X_df_train[singleResponseVariable].values)
    # X_df_train = X_df_train.drop(singleResponseVariable, axis=1)
    # X_df_train.drop('id', axis=1, inplace=True)
    X_df_train = dropBadFiles(X_df_train)
    # print 'Final shape******:' + str(X_df_train.shape)
    answers_1_SINGLE = list(X_df_train[singleResponseVariable].values)
    X_df_train.drop('file', axis=1, inplace=True)
    # X_df_train.drop('patient_id', axis=1, inplace=True)
    print X_df_train.shape
    # X_df_train = X_df_train.loc[X_df_train['file_size'] > 500000]
    print X_df_train.shape
    X_df_train.drop('file_size', axis=1, inplace=True)
    # X_df_train.drop('sequence_id', axis=1, inplace=True)
    X_df_train = X_df_train.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

    # Shuffle rows since they are ordered
    train = X_df_train.iloc[np.random.permutation(len(X_df_train))]
    # train=X_df_train
    # Reset broken index
    train = train.reset_index()

    print("Load test.csv...")
    ieegFeatures = IeegFeatures(test_dir, False)
    df_cols_test = ieegFeatures.ieegGenCols()
    print len(df_cols_test)
    F_NAME_TEST = TEST_FEAT_BASE + TEST_PREFIX_ALL + '-feat_TEST_df.csv'
    X_df_TEST = pandas.read_csv(F_NAME_TEST, engine='python')
    X_df_TEST.drop('Unnamed: 0', axis=1, inplace=True)
    # X_df_TEST.drop('id', axis=1, inplace=True)
    X_df_TEST.drop('file', axis=1, inplace=True)
    # X_df_TEST.drop('patient_id', axis=1, inplace=True)
    # X_df_TEST.drop('file_size', axis=1, inplace=True)
    # X_df_TEST.drop('sequence_id', axis=1, inplace=True)
    test = X_df_TEST

    # train = pre.StandardScaler().fit_transform(train)
    # imp = Imputer()
    # imp.fit(train)
    # train = imp.transform(train)
    #
    # test = pre.StandardScaler().fit_transform(test)
    # imp = Imputer()
    # imp.fit(test)
    # test = imp.transform(test)

    # train = train[rawNames()]

    print("Process tables...")
    # ch2 = SelectKBest(k=50)
    # X_train_features = ch2.fit_transform(train, answers_1_SINGLE)
    # X_train_features = X_df_train_SINGLE[:, b.get_support()]

    # glm_factor = np.asarray(train.columns)[ch2.get_support()]

    # glm_factor=(bestFeaturesPCA(rawNames(),maxFeatures=280))
    glm_factor = (rawNames())

    # X_reduced=train[rawNames()]
    #
    # from sklearn import preprocessing
    # data_scaled = pd.DataFrame(preprocessing.scale(X_reduced), columns=X_reduced.columns)
    #
    # # PCA
    # pca = PCA(n_components=5)
    # pca.fit_transform(data_scaled)
    #
    # # Dump components relations with features:
    # X_reduced=pd.DataFrame(pca.components_, columns=data_scaled.columns, index = ['PC-1','PC-2','PC-3','PC-4','PC-5'])
    #
    #
    # glm_factor = X_reduced.columns
    return train, test, glm_factor

    #
    # # normalize data
    # df_norm = (X_reduced - X_reduced.mean()) / X_reduced.std()
    #
    # svd = TruncatedSVD(n_components=50, random_state=42)
    # pca = PCA(n_components=50)
    # # X_reduced = model.fit_transform(df_norm)
    # X_reduced=pca.fit_transform(df_norm)
    #
    # # X_reduced = TSNE(n_components=20, verbose=2).fit_transform(X_reduced)
    # X_reduced= pd.DataFrame(pca.components_, columns=df_norm.columns)


def rawNames():
    cols = list()
    n = n_16
    for i in range(1, n + 1):
        cols.append('mean_{}'.format(i))
    for i in range(1, n + 1):
        cols.append('median_{}'.format(i))
    for i in range(1, n + 1):
        cols.append('std_{}'.format(i))
    for i in range(1, n + 1):
        cols.append('skew_{}'.format(i))
    for i in range(1, n + 1):
        cols.append('kurt_{}'.format(i))
    for i in range(1, n + 1):
        cols.append('var_{}'.format(i))
    # for i in range(1, n + 1):
    #     cols.append('m6_{}'.format(i))
    # for i in range(1, n + 1):
    #     cols.append('m4_{}'.format(i))
    for i in range(1, n_psd + 1):
        cols.append('psd_{}'.format(i))
    # for i in range(1, n_AR + 1):
    #     cols.append('AR_{}'.format(i))
    for i in range(1, n_corr_coeff + 1):
        cols.append('corcoef_{}'.format(i))
    # for i in range(1, n + 1):
    #     cols.append('hurst_{}'.format(i))
    for i in range(1, n_plv + 1):
        cols.append('plv_{}'.format(i))
    # for i in range(1, n_p_corr + 1):
    #     cols.append('cpc_{}'.format(i))
    return cols


import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def dropBadFiles(df):
    print df.shape
    bad_files = pandas.read_csv("train_and_test_data_labels_safe.csv", engine='python')

    print 'Start shape:' + str(df.shape)
    for index, row in bad_files.iterrows():
        safe = str(row['safe'])  # file name
        if safe == '0':
            f_name = row['image']  # file name
            # print 'droping:' + str(f_name)
            df = df.drop(df[df.file == f_name].index)

    print 'Final shape:' + str(df.shape)
    return df


if __name__ == '__main__':
    print('XGBoost: {}'.format(xgb.__version__))

    import os
    import subprocess
    import sys

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train, test, features = read_test_train()
    print('Length of train: ', len(train))
    print('Length of test: ', len(test))
    print('Features [{}]: {}'.format(len(features), sorted(features)))
    # test_prediction, score = run_single(train, test, features, 'result')
    test_prediction, score = run_kfold(3, train, test, features, singleResponseVariable)
    create_submission(score, test, test_prediction)