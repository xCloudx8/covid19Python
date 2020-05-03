import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import linear_model
from sklearn.metrics import max_error
import math 

def keras_sequential(train_dataset):
 
  model = keras.Sequential([
    layers.Dense(100, activation='linear', 
    input_shape=[len(train_dataset.keys())]),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.Adamax(learning_rate=0.35, beta_1=0.9, beta_2=0.999)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def linearRegressor(X, y, de):
  X = X[de:]
  y = y[de:]

  linearRegr = linear_model.LinearRegression()
  linearRegr.fit(X,y)
  score = linearRegr.score(X, y)

  yPred = linearRegr.predict(X)
  error = max_error(y, yPred)

  Xtest = []
  gp = 40

  for i in range(de, de+gp):
    Xtest.append([i])
  
  yPred_linear = linearRegr.predict(Xtest)

  yPredMax = []
  yPredMin = []

  for i in range(0, len(yPred_linear)):
    yPredMax.append(yPred_linear[i] + error)
    yPredMin.append(yPred_linear[i] - error)
  
  return yPredMax, yPredMin, yPred_linear, Xtest