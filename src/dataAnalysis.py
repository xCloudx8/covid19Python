import pandas as pd
import matplotlib 
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from datetime import timedelta

def dataFrame():
    df = pd.read_csv('data/dataCsv.csv', sep=',', header=0)
    return df

def plotting():
    df = dataFrame()
    df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d')
    df1 = df[['data', 'totale_positivi']]
    df1.plot(kind='scatter', x='data', y='totale_positivi')
        
    #plt.show()

def plottingLinear(dataset, yPred, yPredMax, yPredMin, de, XTest, X, y):
    gp = 40
    data_eff = datetime.strptime(dataset['data'][de], '%Y-%m-%dT%H:%M:%S')
    # date prevision
    date_prev = []
    x_ticks = []
    step = 5
    data_curr = data_eff
    x_current = de
    n = int(gp/step)

    for i in range(0, n):
        date_prev.append(str(data_curr.day) + '/' + str(data_curr.month))
        x_ticks.append(x_current)
        data_curr = data_curr + timedelta(days=step)
        x_current = x_current + step

    plt.grid()
    plt.scatter(X, y, color='black')
    plt.plot(XTest, yPred, color='green', linewidth=2)
    plt.plot(XTest, yPredMax, color='red', linewidth=1, linestyle='dashed')
    plt.plot(XTest, yPredMin, color='red', linewidth=1, linestyle='dashed')
    plt.xlabel('Days')
    plt.xlim(de,de+gp)
    plt.xticks(x_ticks, date_prev)
    plt.ylabel('Epidemics Progression Index (EPI)')
    plt.yscale("log")
    plt.savefig("EPI-prediction.png")
    plt.show()

if __name__ == '__main__':
    plotting()