from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#model api
inputs = keras.Input(shape=(100,))
dense = layers.Dense(51, activation="sigmoid")
x = dense(inputs)
x = layers.Dense(51, activation="sigmoid")(x)
y = layers.Dense(51, activation="linear")(x)
outputs = layers.Dense(1,activation='linear')(y)
regressor = keras.Model(inputs=inputs, outputs=outputs)
regressor.summary()
#end model api

regressor_json = regressor.to_json()

with open('regressor_games.json','w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('regressor_games.h5')

#model save
regressor.save('/home/willi4n/python/Regressao_multiplas_saidsa')
regressorsaved = keras.models.load_model('/home/willi4n/python/Regressao_multiplas_saidsa')
print(regressorsaved)
#end movel save