# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:06:30 2023

@author: Denis
"""
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split 
import pickle
from keras.models import Sequential, model_from_json
from keras.layers import Dense 
from tensorflow.keras import layers




ok1=np.loadtxt('plcclanok/train/oktest/ok1.dat',dtype=(int))
ok2=np.loadtxt('plcclanok/train/oktest/ok2.dat',dtype=(int))
ok3=np.loadtxt('plcclanok/train/oktest/ok3.dat',dtype=(int))
ok4=np.loadtxt('plcclanok/train/oktest/ok4.dat',dtype=(int))
ok5=np.loadtxt('plcclanok/train/oktest/ok5.dat',dtype=(int))

zmena1=np.loadtxt('plcclanok/train/zmenatest/zmena1.dat',dtype=(int))
zmena2=np.loadtxt('plcclanok/train/zmenatest/zmena2.dat',dtype=(int))
zmena3=np.loadtxt('plcclanok/train/zmenatest/zmena3.dat',dtype=(int))
zmena4=np.loadtxt('plcclanok/train/zmenatest/zmena4.dat',dtype=(int))
zmena5=np.loadtxt('plcclanok/train/zmenatest/zmena5.dat',dtype=(int))
#%%

ok=np.row_stack((ok1,ok2,ok3,ok4))#,ok5))
zmena=np.row_stack((zmena1,zmena2,zmena3,zmena4))#,zmena5))

# ok=np.insert(ok,0,22,axis=1)
# zmena=np.insert(zmena,0,33,axis=1)

labok=np.full(len(ok),1)
labzle=np.full(len(zmena),0)


#%%
t=[]
# trainzoradene=np.concatenate((ok[:len(zmena)],zmena))
trainzoradene=np.row_stack((ok,zmena))
labels=np.concatenate((labok,labzle))
# reshape data for CNN
trainzoradene = trainzoradene.reshape((21280, 16, 1))

# convert labels from -1, 1 to 0, 1
labels = (labels + 1) // 2
# train, test, train_label, test_label = train_test_split(trainzoradene,labels,test_size=0.2) 

# for i in range(0,len(zmena)*2):
#     t.append(float(trainzoradene[i]))
# t=np.asarray(t)

# def normalize(arr, t_min, t_max):
#     norm_arr = []
#     diff = t_max - t_min
#     diff_arr = max(arr) - min(arr)   
#     for i in arr:
#         temp = (((i - min(arr))*diff)/diff_arr) + t_min
#         norm_arr.append(temp)
#     return norm_arr

# normt=normalize(trainzoradene, 0, 1)
# normt=np.asarray(normt,dtype='float64')
train, test, train_label, test_label = train_test_split(trainzoradene,labels,test_size=0.2)


 #%%
# Call CNN model 
# model = Sequential()
# model.add(Dense(30, input_shape=(16,), activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# add convolutional layer
# create model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
# model = Sequential()

# # add convolutional layer
# model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(16, 1)))

# # add max pooling layer
# model.add(MaxPooling1D(pool_size=2))

# # add flatten layer
# model.add(Flatten())

# # add dense layer
# model.add(Dense(64, activation='relu'))

# # add output layer
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# define the model architecture
def create_model(kernel_size, filters, pool_size, dense_units):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(16, 1)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the hyperparameters to search over
param_grid = {'kernel_size': [3, 5], 'filters': [32, 64], 'pool_size': [2, 4], 'dense_units': [32, 64]}

# create the Keras classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, verbose=0)

# perform the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train, train_label)

# summarize results
print(f"Best score: {grid_result.best_score_} using {grid_result.best_params_}")
# train model
model = create_model(**grid_result.best_params_)
model.fit(train, train_label, epochs=10, batch_size=32, validation_split=0.2)
# model.fit(train, train_label, epochs=10, validation_split=0.2)
# # compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # evaluate the keras model
# _, accuracy = model.evaluate(test, test_label)
# print('Accuracy: %.2f' % (accuracy*100))


#%%
# load and preprocess test data
# test_data = np.load('test_data.npy')
# test_labels = np.load('test_labels.npy')
# test_data = test_data.reshape((len(test_data), 16, 1))
# test_labels = (test_labels + 1) // 2

# evaluate model on test data
test_loss, test_acc = model.evaluate(test, test_label)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# serialize model to JSON
# model_json = model.to_json()
# with open("plcmodelPRVY.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("plcmodelPRVYvahy.h5")
# print("Saved model to disk")
 
# # later...
 
# # load json and create model
# json_file = open('plcmodelPRVY.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("plcmodelPRVYvahy.h5")
# print("Loaded model from disk")

# testok=np.loadtxt('plcclanok/test/testok4.dat',dtype=(int))
# testzle=np.loadtxt('plcclanok/test/testzmena4.dat',dtype=(int))
# oklabel=np.full(len(testok),1)
# zlelabel=np.full(len(testzle),-1)
# # # evaluate loaded model on test data
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = model.evaluate(testok, oklabel, verbose=0)
# score = model.evaluate(testzle, zlelabel, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))