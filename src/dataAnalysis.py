import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt

def dataFrame():
    df = pd.read_csv('data/dataCsv.csv', sep=',', header=0)
    return df

def plotting():
    df = dataFrame()
    df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d')
    df1 = df[['data', 'totale_positivi']]
    df2 = df[['data', 'nuovi_positivi', 'variazione_totale_positivi']]

    df1.plot(kind='line', x='data')
    df2.plot(kind='line', x='data')
        
    plt.show()