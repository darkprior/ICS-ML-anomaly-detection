# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:33:18 2023

@author: Denis
"""

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import os
import numpy as np
import glob
import matplotlib.pyplot as plt

import sklearn


ok1=np.loadtxt('train/ok_dokopy15/ok1.dat',dtype=(int))
ok2=np.loadtxt('train/ok_dokopy15/ok2.dat',dtype=(int))
ok3=np.loadtxt('train/ok_dokopy15/ok3.dat',dtype=(int))
ok4=np.loadtxt('train/ok_dokopy15/ok4.dat',dtype=(int))
ok5=np.loadtxt('train/ok_dokopy15/ok5.dat',dtype=(int))

zmena1=np.loadtxt('train/zmena_dokopy15/zmena1.dat',dtype=(int))
zmena2=np.loadtxt('train/zmena_dokopy15/zmena2.dat',dtype=(int))
zmena3=np.loadtxt('train/zmena_dokopy15/zmena3.dat',dtype=(int))
zmena4=np.loadtxt('train/zmena_dokopy15/zmena4.dat',dtype=(int))
zmena5=np.loadtxt('train/zmena_dokopy15/zmena5.dat',dtype=(int))
#%%

ok=np.row_stack((ok1,ok2,ok3,ok4,ok5))
zmena=np.row_stack((zmena1,zmena2,zmena3,zmena4,zmena5))

ok=np.concatenate((ok))
zmena=np.concatenate((zmena))

labok=np.full(len(ok),1)
labzle=np.full(len(zmena),-1)


#%%
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

trainzoradene=np.row_stack((ok[:len(zmena)],zmena))

trainzoradene=np.concatenate(trainzoradene)

labels=np.concatenate((labok[:len(zmena)],labzle))
trainzoradene=np.reshape(trainzoradene,(-1,1))
# scaler=RobustScaler()
scaler = StandardScaler()
# scaler= MinMaxScaler() 
# transform data
scaled = scaler.fit_transform(trainzoradene)

# print(scaler.transform(trainzoradene))



train, test, train_label, test_label = train_test_split(trainzoradene,labels,test_size=0.3)
# train=np.reshape(train,(-1,1))
# test=np.reshape(test,(-1,1))
# 1e-5, 1e-4, 1e-3, ..., 1, 10, 100, ...
cecko=1e-3
param_grid = {'C':[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000],'tol':[1,5,10,12,15,17,20,23,25,30,35,40,45,50,60,70,80,100],'max_iter':[1000,5000,10000,20000,30000]
              ,'penalty':['l1','l2']}
grid = GridSearchCV(LinearSVC(),param_grid,refit = True, verbose=2)
# lsvc = LinearSVC(verbose=0,max_iter=10000,C=cecko,tol=15,class_weight='balanced')
# print(lsvc)


# # lsvc.fit(train, train_label)
# grid.fit(train,train_label)
# grid.best_params_
# score = grid.score(test, test_label)
# print("Score: ", score)

# cv_scores = cross_val_score(grid,test , test_label, cv=10)
# print("CV average score: %.2f" % cv_scores.mean())

# ypred = lsvc.predict(test)

# cm = confusion_matrix(test_label, ypred)
# # print(cm)

# cr = classification_report(test_label, ypred)
# print(cr) 

#%%

# testok=np.loadtxt('test/testok1.dat',dtype=(int))
# testok=np.concatenate(testok)
# testzle=np.loadtxt('test/testzmena1.dat',dtype=(int))
# testzle=np.concatenate(testzle)

# testok=np.reshape(testok,(-1,1))
# testzle=np.reshape(testzle,(-1,1))

# oklabel=np.full(len(testok),1)
# zlelabel=np.full(len(testzle),-1) #-1

# # # evaluate loaded model on test data
# # ypred = lsvc.predict(testok)
# # print("ypredtestok: ", np.mean(ypred))

# score1 = grid.score(testzle, zlelabel)
# score2 = grid.score(testok, oklabel)
# print("OK Score test: ", score2)
# print("ZLE Score test: ", score1)

