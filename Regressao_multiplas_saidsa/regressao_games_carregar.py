import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

arquivo_json = open('regressor_games.json','r')
estrutura_de_rede = arquivo_json.read()
arquivo_json.close()

estrutura_regressor = model_from_json(estrutura_de_rede)
estrutura_regressor.load_weights('regressor_games.h5')

base = pd.read_csv('games.csv')

base = base.dropna(axis=0)

# working with values that are minors than 1
base = base.loc[base['Global_Sales'] < 0.75 ]
base = base.loc[base['Global_Sales'] > 0.5 ]

#save name collumn to compare the price later
nome_jogos = base.Name

#Removing columns
base = base.drop(labels='Name', axis=1)
base = base.drop(labels='Developer', axis=1)
base = base.drop(labels='NA_Sales', axis=1)
base = base.drop(labels='EU_Sales', axis=1)
base = base.drop(labels='JP_Sales', axis=1)
base = base.drop(labels='Other_Sales', axis=1)

# classe
vendas_global = base.iloc[:,4].values# todos os valores sao num, entao n vai precisar da conversao transform

#Criando os previsores
previsores = base.iloc[:,[0,1,2,3,5,6,7,8,9]].values

label = LabelEncoder()

previsores[:,0] = label.fit_transform(previsores[:,0])
previsores[:,2] = label.fit_transform(previsores[:,2])
previsores[:,3] = label.fit_transform(previsores[:,3])
previsores[:,8] = label.fit_transform(previsores[:,8])

index = [0,2,3,8]

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), index )],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# 
camada_entrada = Input(shape=(100,))

camada_oculta1 = Dense(units = 50, activation = 'sigmoid') (camada_entrada)
camada_oculta2 = Dense(units = 50, activation = 'sigmoid') (camada_oculta1)
saida_venda_global = Dense(units = 1, activation = 'linear') (camada_oculta2)
regressor_teste = Model(inputs = camada_entrada, outputs = saida_venda_global)
regressor_teste.compile(optimizer='adam',loss='mse')

regressor_teste.fit(previsores, vendas_global, epochs = 5000, batch_size = 100)
previsao_vendas_global = regressor_teste.predict(previsores)

# Tring in a different way to get better results
estrutura_regressor.compile(optimizer='sgd',loss='mse')

estrutura_regressor.fit(previsores, vendas_global, epochs = 1000, batch_size = 100)
previsao_vendas_global = estrutura_regressor.predict(previsores)