# import IeegFeatures

from IeegFeatures import *
# from IeegConsts import *

from datetime import datetime
import os
import time
import glob
import re
from multiprocessing import Process
import xgboost as xgb

class IeegFeatureGenerator(object):
    """
        Class to generate features for the patients
        @author Solomomk
    """
    def __init__(self, baseDir, isTrain=True):
        now = datetime.now()
        print 'Starting:' + 'IeegFeatureGenerator:' + str(now)
        self.isTrain=isTrain
        self.baseDir=baseDir
        self.allDataFiles=None
        self.cols = None
        self.fullPath=None

        print 'isTrain:' + str(self.isTrain)
        print 'baseDir:' + self.baseDir


    def buildFeatures(self):
        now = datetime.now()
        print 'Starting:' + 'buildFeatures:' +str(now)
        ieegFeatures= IeegFeatures(self.baseDir,self.isTrain)
        self.allDataFiles = ieegFeatures.ieegAllFilesList()

        print 'Total files:' + str(len(self.allDataFiles))

        # if self.isTrain:
            # unique, counts = np.unique(self.allDataFiles['class'], return_counts=True)
            # occurrences = dict(zip(unique, counts))
            # # Count the occurrences of Interictal and Preictal classes
            # print('Interictal samples:', occurrences.get(INTERICTAL))
            # print('Preictal samples:', occurrences.get(PREICTAL))
            #
            # data_random_interictal = np.random.choice(self.allDataFiles[self.allDataFiles['class'] == 0],
            #                                           size=occurrences.get(INTERICTAL))
            # data_random_preictal = np.random.choice(self.allDataFiles[self.allDataFiles['class'] == 1],
            #                                         size=occurrences.get(PREICTAL))
            # data_files = np.concatenate([data_random_interictal, data_random_preictal])
            # data_files.dtype = self.allDataFiles.dtype
            # self.allDataFiles = data_files

        np.random.shuffle(self.allDataFiles)
        print(self.allDataFiles.shape, self.allDataFiles.size)
        # print(self.allDataFiles)


        self.cols=ieegFeatures.ieegGenCols()
        X = ieegFeatures.ieegProcessAllFilesAsDF(self.allDataFiles['file'])
        print('X_shape:', X.shape, 'X_size:', X.size)
        X_df=pandas.DataFrame(X, columns=self.cols)
        X_df.drop(X_df.head(1).index, inplace=True)

        if self.isTrain:
            self.fullPath=TRAIN_FEAT_BASE + TRAIN_PREFIX_ALL +'-feat_TRAIN_df.csv'
            X_df[singleResponseVariable] = X_df[singleResponseVariable].astype('category')

        else:
            self.fullPath = TEST_FEAT_BASE + TEST_PREFIX_ALL + '-feat_TEST_df.csv'

        print str(X_df.shape)
        print 'Writing:' + self.fullPath
        X_df.to_csv(self.fullPath , sep=',')
        print X_df.head(1)
        # return X_df

if __name__ == '__main__':
    print('XGBoost: {}'.format(xgb.__version__))
    train_dir=TRAIN_DATA_FOLDER_IN_ALL
    test_dir=TEST_DATA_FOLDER_IN_ALL

    # ieegFeatureGeneratorTrain = IeegFeatureGenerator(train_dir, True)
    ieegFeatureGeneratorTest = IeegFeatureGenerator(test_dir, False)

    # ieegFeatureGeneratorTrain.buildFeatures()
    ieegFeatureGeneratorTest.buildFeatures()

    # if 1:
    #     # p = dict()
    #     p1 = Process(target=ieegFeatureGeneratorTrain.buildFeatures())
    #     p2 = Process(target=ieegFeatureGeneratorTest.buildFeatures())
    #
    #     # p1.daemon = True
    #     p1.start()
    #
    #     # p2.daemon = True
    #     p2.start()
    #
    #     p1.join()
    #     p2.join()

print 'done'

# X_df=pandas.read_csv(ieegFeatureGeneratorTrain.fullPath, engine='python')


