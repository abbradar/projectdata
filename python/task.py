#!/usr/bin/env python3

import numpy as np
import pandas
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import matplotlib
import xgboost

def pandas_wrap(op, arr):
  return pandas.DataFrame(op(arr), index=arr.index, columns=arr.columns)

np.random.seed(42)
matplotlib.use("QT5Agg")
from matplotlib import pyplot as plt

fullset = pandas.read_csv("train.csv").drop("ID", axis=1)
fullresult = fullset["TARGET"]
fullset.drop("TARGET", axis=1, inplace=True)

# Normalize
fullset = fullset.apply(lambda x: x - x.min() + 1)

# Seemingly arbitrary ¯\_(ツ)_/¯
LOGSCALE_BORDER = 1e4
# Ugh... ternaries in Python
fullset = fullset.apply(lambda x: np.log10(x) if x.max() > LOGSCALE_BORDER else x)

fullset = pandas_wrap(preprocessing.scale, fullset)

trainset, testset, trainresult, testresult = train_test_split(fullset, fullresult, test_size=0.25)

classifier = xgboost.XGBClassifier(max_depth=3)

model = classifier.fit(trainset, trainresult)
probas = model.predict_proba(testset)
fpr, tpr, thresholds = roc_curve(testresult, probas[:, 1])
result_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.show()

evalset = pandas.read_csv("test.csv")
evalresult = evalset[['ID']]
evalset = evalset.drop('ID', axis=1)

evalresult.insert(1, 'TARGET', model.predict(evalset))
evalresult.to_csv('result.csv', index=False)