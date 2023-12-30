# Stock recommandation via machine learning algorithms
An stock investment assistant tool which utilized supervised machine learning models such as Logistic Regression, Random Forest, and Support Vector Machine to predict the stock’s 60 days’ return rate. If a specific stock outperformed the average return rate, the model would recommend to hold.

# Tools
- [Sklearn](https://scikit-learn.org/stable/)
- [Tushare](https://tushare.pro/) API
- [multiprocessing-python based parallelism](https://docs.python.org/3/library/multiprocessing.html)
# How to use?
- Step1: Run [0-Get Fina Indicators Data.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/0-Get%20Fina%20Indicators%20Data.py) to get financial indicator data from 2019-Q1 to 2020-Q1.
- Step 2: Calculate cumulated ROI for each stock, imputation, train test split by running [1-Dataset Construction.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/1-Dataset%20Construction.py).
- Step 3: Get the statistics of features by running [2-Dataset Description.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/2-Dataset%20Description.py)
- Step 3: Train machine learning models
  - [3-LG.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/3-LG.py)
  - [3-RF.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/3-RF.py)
  - [3-SVM.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/3-SVM.py)
- Step 4: Testing via [4-Testing.py](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/4-Testing.py)
# Jupyter Notebooks for fine-tunning
- [LG Fine Tunning-checkpoint.ipynb](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/LG%20Fine%20Tunning-checkpoint.ipynb)
- [RF Fine Tunning-checkpoint.ipynb](https://github.com/WideSu/Stock-recommandation-via-machine-learning-algorithms/blob/main/RF-checkpoint.ipynb)

