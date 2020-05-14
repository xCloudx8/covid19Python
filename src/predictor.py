from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

'''
    Auxiliary functions
'''
def dataFrame():
    df = pd.read_csv('data/dataCsv.csv', sep=',', header=0)
    return df

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=3).mean()
    rolstd = pd.Series(timeseries).rolling(window=3).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def seasonal_dec(ts_log):
    decomposition = seasonal_decompose(ts_log, period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    '''
    plt.subplot(411)
    plt.plot(ts_log, label= 'Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label = 'Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.tight_layout()
    '''

    ts_log_decompose =  residual
    ts_log_decompose.dropna(inplace=True)
    test_stationarity(ts_log_decompose)

def forecasting(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=12)
    lag_pacf = pacf(ts_log_diff, nlags = 12, method = 'ols')

    #Plot Acf
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

def ARmodel(ts_log, ts_log_diff):
    model = ARIMA(ts_log,order=(2,1,0))
    results_ar = model.fit(disp=-1)
    plt.plot(ts_log_diff)
    plt.plot(results_ar.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results_ar.fittedvalues - ts_log_diff)**2))
    plt.show()
    return results_ar.fittedvalues
    
def predict():
    df = dataFrame()
    con = df['data']
    df['data']=pd.to_datetime(df['data'])
    df.set_index('data', inplace= True)
    ts = df['totale_positivi']
    #test_stationarity(ts)

    ts_log = np.log(ts)
    moving_avg = pd.Series(ts_log).rolling(window=3).mean()
    
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)
    
    expwighted_avg = pd.Series(ts_log).ewm( halflife = 3).mean()
    
    ts_log_ewma_diff = ts_log - expwighted_avg
    
    ts_log_diff = ts_log - ts_log.shift()
    
    ts_log_diff.dropna(inplace = True)
    
    seasonal_dec(ts_log)
    forecasting(ts_log_diff)
    arimaFitted = ARmodel(ts_log, ts_log_diff)

    prediction_ARIMA_diff = pd.Series(arimaFitted, copy= True)
    predictions_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
    prediction_ARIMA_LOG = pd.Series(ts_log[0], index = ts_log.index)
    prediction_ARIMA_LOG = prediction_ARIMA_LOG.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    prediction_ARIMA = np.exp(prediction_ARIMA_LOG)
    plt.close()
    plt.plot(ts)
    plt.plot(prediction_ARIMA)
    plt.title('RMSE: %4.f'% np.sqrt(sum((prediction_ARIMA-ts)**2)/len(ts)))
    plt.show()
    pass

if __name__ == '__main__':
    predict()