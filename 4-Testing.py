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

# Loading holdout dataset
holdout_dataset = pd.read_excel('holdout_dataset.xlsx')
# If error, please get rid of the folloing line: holdout_dataset.set_index(["Unnamed: 0"], inplace=True)
holdout_dataset.set_index(["Unnamed: 0"], inplace=True)
holdout_dataset = holdout_dataset.T
for i in holdout_dataset.columns:
    if len(str(i)) == 1:
        holdout_dataset.rename(columns={i:'00000'+str(i)}, inplace=True)
    if len(str(i)) == 2:
        holdout_dataset.rename(columns={i:'0000'+str(i)}, inplace=True)
    if len(str(i)) == 3:
        holdout_dataset.rename(columns={i:'000'+str(i)}, inplace=True)
    if len(str(i)) == 4:
        holdout_dataset.rename(columns={i:'00'+str(i)}, inplace=True)
    if len(str(i)) == 5:
        holdout_dataset.rename(columns={i:'0'+str(i)}, inplace=True)
    if len(str(i)) == 6:
        holdout_dataset.rename(columns={i:str(i)}, inplace=True)
        
# Loading training dataset
training_dataset = pd.read_excel('training_dataset.xlsx')
# If error, please get rid of the folloing line: training_dataset.set_index(["Unnamed: 0"], inplace=True)
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
# use the holdout dataset as testing dataset
X_test = holdout_dataset.T.iloc[:, 1:106].values
Y_test = holdout_dataset.T.iloc[:, 0].values

# scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# fitting RandomForest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=0,max_depth=10,max_features='auto',min_samples_leaf=4,min_samples_split=10,n_estimators=133)
classifier.fit(X_train, Y_train)

# predicting 
Y_pred = classifier.predict(X_test)

result = pd.DataFrame(columns=holdout_dataset.columns).T
result['pred_result(1=Investment)'] = pd.Series(Y_pred).values

print(result)
result.to_excel('pred_result.xlsx')
