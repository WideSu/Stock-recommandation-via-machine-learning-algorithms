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

# fitting LG to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
print('LogisticRegression Training Accuracy（Before）: ', classifier.score(X_train, Y_train))

# predicting 
Y_pred = classifier.predict(X_test)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print('LogisticRegression Confusion Matrix (Before): ',cm)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
print('LogisticRegression Testing Accuracy (Before): ',accuracy)


# GridSearchCV to tune
from sklearn.model_selection import GridSearchCV
C = list(np.power(10.0, np.arange(-10,10)))
# max_iter = [1, 10, 100, 500]
# class_weight = ['balanced', None]
# solver = ['liblinear','sag','lbfgs','newton-cg']

parameters = {'C':C,
#           'max_iter':max_iter,
#           'class_weight':class_weight,
#           'solver':solver
             }

grid_search = GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                          cv = 4,n_jobs = -1)

grid_search = grid_search.fit(X_train,Y_train)
accuracy_after = grid_search.best_score_

# GridSerchCV results
print('LogisticRegression Best Params: ',grid_search.best_params_)
print('LogisticRegression Training Accuracy (After): ',accuracy_after)

classifier = LogisticRegression(
#     class_weight=grid_search.best_params_['class_weight'], 
     C=grid_search.best_params_['C'],
#     max_iter=grid_search.best_params_['max_iter'],
#     solver=grid_search.best_params_['solver']
                                )
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_after = accuracy_score(Y_test,Y_pred)
cm_after = confusion_matrix(Y_test,Y_pred)
print('LG Confusion Matrix (After): ',cm_after)
print('LG Testing Accuracy (After): ',accuracy_after)


# draw pic


C = list(np.power(10.0, np.arange(-10,10)))
accuracy_train = []
accuracy_test = []

for i in C:
    classifier = LogisticRegression(C=i,random_state=0)
    classifier.fit(X_train,Y_train)
    accuracy_train.append(classifier.score(X_train, Y_train))
    
    
    Y_pred = classifier.predict(X_test)
    accuracy_test.append(accuracy_score(Y_test,Y_pred))  

plt.figure(figsize=(6,4))
ax = plt.gca()
X_axis = C
ax.set_ylim(0.575,0.68)
ax.set_xlim(min(C),max(C),1)

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel('parameters_C',size=16)
plt.ylabel('AUC',size=16)

x = np.arange(len(C)).tolist()

plt.semilogx(C,accuracy_train, color='black',alpha=0.8,label='AUC_train',linestyle='--')
plt.semilogx(C,accuracy_test,color='black',alpha=0.8,label='AUC_test')
plt.axvline(x = C[accuracy_test.index(max(accuracy_test))],ymin=0,ymax=max(accuracy_test),color='black',alpha=0.8,linestyle='--')
plt.scatter(x = C[accuracy_test.index(max(accuracy_test))],y=max(accuracy_test),color='red',marker='.',alpha=0.4,s=500)
plt.legend(loc='best')
plt.show()