import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from . import models

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def predict():
    df = pd.read_csv('data/dataCsv.csv', sep=',')
    #df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d')

    #Clean dataset
    df = df.drop(['data', 'stato', 'note_en', 'note_it', 'casi_testati'], axis=1)
    
    #print(data.isna().sum())

    #Convert into categorical columns if exists
    
    #Datasets
    train_dataset = df.sample(frac=0.8,random_state=0)
    test_dataset = df.drop(train_dataset.index)

    #Plot with seaborn
    #test_dataset.plot(kind='scatter', x='data', y='totale_positivi')
    #plt.show()

    train_stats = train_dataset.describe()
    train_stats.pop("totale_positivi")
    train_stats = train_stats.transpose()
    
    #Split features from labels
    train_labels = train_dataset.pop('totale_positivi')
    test_labels = test_dataset.pop('totale_positivi')

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    #Use model
    model = models.keras_sequential(train_dataset)
    #print(model.summary())
    
    #Train model
    EPOCHS = 1000
    history = model.fit(
        normed_train_data,
        train_labels,
        epochs = EPOCHS,
        validation_split = 0.3,
        verbose = 0,
        callbacks=[tfdocs.modeling.EpochDots()]
    )
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    #Visualize data
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    print(hist)
if __name__ == '__main__':
    predict()