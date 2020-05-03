import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from . import models
from src import dataAnalysis
from sklearn import linear_model

def predict():
    df = dataAnalysis.dataFrame()
    total_cases = df['totale_casi']
    total_analysis = df['tamponi']

    y = []
    ta_increase = []

    for i in range(1,len(total_analysis)):
        actEPI = (total_cases[i] - total_cases[i-1])/ (total_analysis[i] - total_analysis[i-1])*100
        ta_increase.append(total_analysis[i]-total_analysis[i-1])
        y.append(actEPI)
    
    X = []
    for i in range(1, len(y)+1):
        X.append([i])

    de = 14 + 7
    
    yPredMax, yPredMin, yPred_linear, Xtest = models.linearRegressor(X, y, de)

    X = X[de:]
    y = y[de:]
    dataAnalysis.plottingLinear(df, yPred_linear, yPredMax, yPredMin, de, Xtest, X, y)

if __name__ == '__main__':
    pass