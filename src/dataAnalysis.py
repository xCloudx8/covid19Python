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
    plt.close()    
    #plt.show()

if __name__ == '__main__':
    plotting()