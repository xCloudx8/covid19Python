import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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