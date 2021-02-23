import pandas as pd 
import numpy as np
import sys
import xlwt
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Loading dataset
training_dataset = pd.read_excel('training_dataset.xlsx')

# If error, please get rid of folloing line: training_dataset.set_index(["Unnamed: 0"], inplace=True)
training_dataset.set_index(["Unnamed: 0"], inplace=True)
training_dataset = training_dataset.T

for i in training_dataset.columns:
    if len(str(i)) == 1:
        training_dataset.rename(columns={i:'00000'+str(i)}, inplace=True)
    if len(str(i)) == 2:
        training_dataset.rename(columns={i:'0000'+str(i)}, inplace=True)
    if len(str(i)) == 3:
        training_dataset.rename(columns={i:'000'+str(i)}, inplace=True)
    if len(str(i)) == 4:
        training_dataset.rename(columns={i:'00'+str(i)}, inplace=True)
    if len(str(i)) == 5:
        training_dataset.rename(columns={i:'0'+str(i)}, inplace=True)
    if len(str(i)) == 6:
        training_dataset.rename(columns={i:str(i)}, inplace=True)


# split the data into independent 'X' and dependent 'Y' variables
X = training_dataset.T.iloc[:, 1:106].values
Y = training_dataset.T.iloc[:, 0].values
# split the dataset into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0)
classifier.fit(X_train, Y_train)
print('RandomForest Training Accuracy: ', classifier.score(X_train, Y_train))

# predicting 
Y_pred = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print('RandomForest Confusion Matrix (Before): ',cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
print('RandomForest Testing Accuracy (Before): ',accuracy)


                          
# 参数设置
from sklearn.model_selection import GridSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 0, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]


# Create the parameters grid
parameters = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
             }

grid_search = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv = 5,n_jobs = -1)

grid_search = grid_search.fit(X_train,Y_train)
accuracy_after = grid_search.best_score_

              
# GridSerchCV results
print('RandomForest Best Params: ',grid_search.best_params_)
print('RandomForest Training Accuracy (After): ',accuracy_after)

classifier = RandomForestClassifier(max_depth=grid_search.best_params_['max_depth'],
                                    max_features=grid_search.best_params_['max_features'],
                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                    min_samples_split=grid_search.best_params_['min_samples_split'],
                                    n_estimators=grid_search.best_params_['n_estimators'])

classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_after = accuracy_score(Y_test,Y_pred)
cm_after = confusion_matrix(Y_test,Y_pred)
print('RandomForest Confusion Matrix (After): ',cm_after)
print('RandomForest Testing Accuracy (After): ',accuracy_after)