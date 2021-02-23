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
# fitting kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(random_state=0)
classifier.fit(X_train,Y_train)
print('SVC Training Accuracy: ', classifier.score(X_train, Y_train))

# predicting 
Y_pred = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print('SVC Confusion Matrix (Before): ',cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
print('SVC Testing Accuracy (Before): ',accuracy)

                          
# Parameters set
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100,1000],'kernel':['linear']},
             {'C':[1,10,100,1000],'kernel':['rbf']}]

grid_search = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv = 5,
                          n_jobs= -1)

grid_search = grid_search.fit(X_train,Y_train)
accuracy_after = grid_search.best_score_

# GridSerchCV results
print('SVC Best Params: ',grid_search.best_params_)
print('SVC Training Accuracy (After): ',accuracy_after)


classifier = SVC(kernel=grid_search.best_params_['kernel'], C=grid_search.best_params_['C'])
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_after = accuracy_score(Y_test,Y_pred)
cm_after = confusion_matrix(Y_test,Y_pred)
print('SVC Confusion Matrix (After): ',cm_after)
print('SVC Testing Accuracy (After): ',accuracy_after)
