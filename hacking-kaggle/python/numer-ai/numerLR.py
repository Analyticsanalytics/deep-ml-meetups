from __future__ import print_function
from __future__ import division

import math
import time
import random
random.seed(67)

import numpy as np
np.random.seed(712)

import pandas as pd

# from sklearn.decomposition import RandomizedPCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import log_loss
# from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# from sklearn.feature_selection import SelectKBest
# from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
# from sklearn.manifold import Isomap
# from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
# from sklearn.decomposition import TruncatedSVD
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA, IncrementalPCA
# import sklearn.linear_model as slm

from numerCommonML import *
from keras.models import Sequential
from keras.layers import Dense

class NumerLR(object):
    def __init__(self, baseDir):
        # now = datetime.now()
        print('Starting:' + 'NumerLR:')
        self.baseDir = baseDir

    # @NumerCommonML.fn_timer
    def run(self):
        # load data
        # X_test, X_train, X_valid, df_test, y_train, y_valid = NumerCommonML.loadFeaturesSplit()


        final_scores={}

        clfs = {
            'NumerCommon.PIP_LR':NumerCommonML.PIP_LR,
            # 'NumerCommon.PIP_KNN':NumerCommonML.PIP_KNN,
            # 'NumerCommon.PIP_LR_MAX_SCALER':NumerCommonML.PIP_LR_MAX_SCALER,
            'NumerCommon.PIP_BEST':NumerCommonML.PIP_BEST,
            'NumerCommon.PIP_CALIBRATED_LR_ISOTONIC_1':NumerCommonML.PIP_CALIBRATED_LR_ISOTONIC_1,
            'NumerCommon.PIP_CALIBRATED_LR_ISOTONIC_2':NumerCommonML.PIP_CALIBRATED_LR_ISOTONIC_2,
            'NumerCommon.PIP_CALIBRATED_LR_ISOTONIC_3':NumerCommonML.PIP_CALIBRATED_LR_ISOTONIC_3,

            'NumerCommon.PIP_CALIBRATED_LR_SIGMOID_1':NumerCommonML.PIP_CALIBRATED_LR_SIGMOID_1,
            'NumerCommon.PIP_CALIBRATED_LR_SIGMOID_2':NumerCommonML.PIP_CALIBRATED_LR_SIGMOID_2,
            'NumerCommon.PIP_CALIBRATED_LR_SIGMOID_3':NumerCommonML.PIP_CALIBRATED_LR_SIGMOID_3,
            # 'NumerCommon.PIP_CALIBRATED_LR_SIGMOID_SGD':NumerCommonML.PIP_CALIBRATED_LR_SIGMOID_SGD,
            # 'NumerCommon.PIP_CALIBRATED_LR_SIGMOID_SGD2':NumerCommonML.PIP_CALIBRATED_LR_SIGMOID_SGD2,
            # 'NumerCommon.PIP_CALIBRATED_RF':NumerCommonML.PIP_CALIBRATED_RF,
            # 'NumerCommon.PIP_GRID_LR':NumerCommonML.PIP_GRID_LR,
            # 'NumerCommon.PIP_GRID_LR_PURE':NumerCommonML.PIP_GRID_LR_PURE,
        }




        # X_test, X_train, X_valid, df_test, y_train, y_valid = NumerCommonML.loadDataSplit(r_state=989)
        #
        # # 2-class logistic regression in Keras
        # model = Sequential()
        # model.add(Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
        # model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        #
        #
        # model.fit(X_train, y_train, nb_epoch=10, validation_data=(X_valid, y_valid))
        # scores = model.evaluate(X_valid, y_valid, verbose=0)
        # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        i = 0
        i = i + 1
        for key, classifier in clfs.iteritems():
            r_state = 87 * 14 * 36 * i
            print ('Random State:' + str(r_state))
            X_test, X_train, X_valid, df_test, y_train, y_valid = NumerCommonML.loadFeaturesSplit(r_state=r_state)

            print('Fitting:' + key)
            print(str(classifier))

            # scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5, scoring='roc_auc')
            # print("ROC AUC: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std()))


            start_time = time.time()
            classifier.fit(X_train, y_train)
            print('Fit: {}s'.format(time.time() - start_time))

            p_valid = classifier.predict_proba(X_valid)
            loss = log_loss(y_valid, p_valid)
            print('log loss: {}'.format(loss))

            p_test = classifier.predict_proba(X_test)
            df_pred = pd.DataFrame({
                'id': df_test['id'],
                'probability': p_test[:, 1]
            })
            NumerCommonML.savePred(df_pred, key, loss)
            final_scores[key]=loss

        #

        print (sorted(final_scores.items(), key=lambda x: x[1]))

if __name__ == '__main__':

    # BASE_FOLDER = '/root/sharedfolder/kaggle/numerai/data/'
    numerLR= NumerLR(baseDir=BASE_FOLDER)
    # NumerCommonML.makeFeatures(BASE_FOLDER=BASE_FOLDER)
    numerLR.run()
