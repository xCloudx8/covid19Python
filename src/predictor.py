import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(50, activation='relu', 
    input_shape=[len(train_dataset.keys())]),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def predict():
    df = pd.read_csv('data/dataCsv.csv', sep=',')
    #df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d')

    #Clean dataset
    df = df.drop(['data', 'stato', 'note_en', 'note_it'], axis=1)
    
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
    #test_labels = test_dataset.pop('totale_positivi')

    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    #Use model
    model = build_model(train_dataset)
    print(model.summary())
    
    #Train model
    EPOCHS = 1000
    history = model.fit(
        normed_train_data,
        train_labels,
        epochs = EPOCHS,
        validation_split = 0.2,
        verbose = 0,
        callbacks=[tfdocs.modeling.EpochDots()]
    )

    #Visualize data
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

if __name__ == '__main__':
    predict()