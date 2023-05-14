# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:02:49 2023

@author: Denis
"""

import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import csv
from itertools import zip_longest 
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, classification_report
import pickle


#%% premenovanie suborov

# folder = r'online_monitoring\extrah\\'
# count = 1
# # count increase by 1 in each iteration
# # iterate all files from a directory
# for file_name in os.listdir(folder):
#     # Construct old file name
#     source = folder + file_name

#     # Adding the count to the new file name and extension
#     destination = folder + "monitor_" + str(count) + ".txt"

#     # Renaming the file
#     os.rename(source, destination)
#     count += 1


#%%

hex_arrNORM=[]
hex_arrANOM=[]
hex_arrMONI=[]
for i in range(1,51):
  with open('normal_data/extrah/paket_'+str(i)+'.txt', 'r') as file:
    hex_str = file.read().strip().replace('\n', '')
    hex_arrNORM.append(np.array(list(bytes.fromhex(hex_str))))


for i in range(1,6):
  with open('modifikacia/extrah/mod'+str(i)+'.txt', 'r') as file:
    hex_str = file.read().strip().replace('\n', '')
    hex_arrANOM.append(np.array(list(bytes.fromhex(hex_str))))
    
for i in range(1,31):
  with open('online_monitoring/extrah/monitor_'+str(i)+'.txt', 'r') as file:
    hex_str = file.read().strip().replace('\n', '')
    hex_arrMONI.append(np.array(list(bytes.fromhex(hex_str))))



arrNORM = np.array([np.pad(row, (0, max(map(len, hex_arrNORM)) - len(row))) for row in hex_arrNORM])
arrMONI = np.array([np.pad(row, (0, max(map(len, hex_arrMONI)) - len(row))) for row in hex_arrMONI])
arrANOM = np.array([np.pad(row, (0, max(map(len, hex_arrANOM)) - len(row))) for row in hex_arrANOM])

# Reshape arrays and concatenate them
reshaped_arr1 = arrNORM.reshape((arrNORM.shape[0], -1))
reshaped_arr2 = arrMONI.reshape((arrMONI.shape[0], -1))
reshaped_arr3 = arrANOM.reshape((arrANOM.shape[0], -1))
diff = abs(reshaped_arr1.shape[1] - reshaped_arr2.shape[1])

# Pad the smaller array with zeros along axis 1
if reshaped_arr1.shape[1] < reshaped_arr2.shape[1]:
    NEWARR1 = np.pad(reshaped_arr1, [(0, 0), (0, diff)], mode='constant')
else:
    reshaped_arr2 = np.pad(reshaped_arr2, [(0, 0), (0, diff)], mode='constant')

diff = abs(NEWARR1.shape[1] - reshaped_arr3.shape[1])

# # Pad the smaller array with zeros along axis 1
if NEWARR1.shape[1] > reshaped_arr3.shape[1]:
    NEWARR1 = np.pad(NEWARR1, [(0, 0), (0, diff)], mode='constant')
else:
    reshaped_arr3 = np.pad(reshaped_arr3, [(0, 0), (0, diff)], mode='constant')

NEWARR3=reshaped_arr3[:,:5585]
# Concatenate the arrays along axis 1
concatenated_array = np.vstack((NEWARR1, reshaped_arr2,NEWARR3))

for i in range(0,len(concatenated_array)):
    for j in range(0,len(concatenated_array[0,:])):
        if concatenated_array[i][j] < 132:
            concatenated_array[i][j]=0
        if concatenated_array[i][j] > 132 and concatenated_array[i][j] <250:
            concatenated_array[i][j]=0
# # Create CSV file and write data to it
with open('example.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(concatenated_array.tolist())

#%%

# # Load data from csv file
# data = pd.read_csv('example.csv')

# # Extract numerical features
# features = data.select_dtypes(include=['float64', 'int64'])

# # Split data into training and testing sets
# train = features.sample(frac=0.8, random_state=1)
# test = features.drop(train.index)
# scorer = make_scorer(mean_squared_error, greater_is_better=False)
# # Hyperparameter tuning with GridSearchCV
# param_grid = {'n_estimators': [10,25,50, 100, 200,500,1000],
#               'max_samples': [0.5, 0.75, 1.0],
#               'contamination': ['auto',0.01, 0.05, 0.1]}
# grid_search = GridSearchCV(IsolationForest(random_state=1), param_grid, cv=5)
# grid_search.fit(train) 


# # Train model with best hyperparameters
# model = IsolationForest(n_estimators=grid_search.best_params_['n_estimators'],
#                         max_samples=grid_search.best_params_['max_samples'],
#                         contamination=grid_search.best_params_['contamination'])
# model.fit(train)


# Load data from csv file
data = pd.read_csv('example.csv')

# Extract numerical features
features = data.select_dtypes(include=['float64', 'int64'])

# Split data into training and testing sets
train = features.sample(frac=0.8, random_state=1)
test = features.drop(train.index)


from sklearn.ensemble import IsolationForest
import numpy as np

class CustomIsolationForest(IsolationForest):
    
    def score(self, X, y=None):
        y_pred = self.predict(X)
        mask = y_pred == -1
        X_recon = np.copy(X)
        X_recon[mask] = np.mean(X[y_pred == 1], axis=0)
        mse = mean_squared_error(X, X_recon)
        return -mse



# # Define custom scorer function
# def custom_scorer(y_pred):
#     try:
#         mse = mean_squared_error(train, y_pred)
#         return mse
#     except:
#         return float('nan')


# Hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [10,25,50, 100, 200,500,1000],
              'max_samples': [0.5, 0.75, 1.0],
              'contamination': ['auto',0.01, 0.05, 0.1]}

grid_search = GridSearchCV(CustomIsolationForest(random_state=1), param_grid, cv=5)
grid_search.fit(train)

# Train model with best hyperparameters
model = IsolationForest(n_estimators=grid_search.best_params_['n_estimators'],
                        max_samples=grid_search.best_params_['max_samples'],
                        contamination=grid_search.best_params_['contamination'])
model.fit(train)


#%%


# Save the trained model object to a file
with open('isolation_forest_model.pkl', 'wb') as f:
    pickle.dump({'model': model}, f)



    
# Load the saved model object from a file
# with open('isolation_forest_model.pkl', 'rb') as f:
#     saved_model = pickle.load(f)

# Extract the model instance, history, and metrics
# loaded_model = saved_model['model']
# loaded_history = saved_model['history']
# loaded_metrics = saved_model['metrics']

#%%

# Predict anomalies on test set
anomalies = model.predict(test)

# Print number of anomalies detected
print("Number of anomalies detected:", np.sum(anomalies == -1))
# Make predictions on the testing set
# y_pred = model.predict(test)

# # Compute and print classification metrics
# print(classification_report(test, y_pred))
