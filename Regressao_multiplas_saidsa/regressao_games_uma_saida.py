import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Data load
base = pd.read_csv('games.csv')
#Check columns name to make easier
columns = base.columns.to_list()

#save name collumn to compare the price later
nome_jogos = base.Name

chk = base.loc[base['Global_Sales'] < 1]

#Removing columns
base = base.drop(labels='Name', axis=1)
base = base.drop(labels='Developer', axis=1)
base = base.drop(labels='NA_Sales', axis=1)
base = base.drop(labels='EU_Sales', axis=1)
base = base.drop(labels='JP_Sales', axis=1)
base = base.drop(labels='Other_Sales', axis=1)

"""
Esta e uma forma de testar a rede
"""

# working with values that are minors than 1
base = base.loc[base['Global_Sales'] < 1 ]
base = base.loc[base['Global_Sales'] > 0.5 ]


# removing lines with no values
base = base.dropna(axis=0)

#create classe
vendas_global_sales = base.iloc[:,4].values# todos os valores sao num, entao n vai precisar da conversao transform

#Criando os previsores
previsores = base.iloc[:,[0,1,2,3,5,6,7,8,9]].values

#remove string values

label = LabelEncoder()

previsores[:,0] = label.fit_transform(previsores[:,0])
previsores[:,2] = label.fit_transform(previsores[:,2])
previsores[:,3] = label.fit_transform(previsores[:,3])
previsores[:,8] = label.fit_transform(previsores[:,8])

index = [0,2,3,8]

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), index )],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# neural network

camada_entrada = Input(shape=(118,))

#ativacao = Activation(activation = 'sigmoid')

camada_oculta1 = Dense(units=59, activation= 'sigmoid') (camada_entrada)
camada_oculta2 = Dense(units=59, activation= 'sigmoid') (camada_oculta1)

camada_saida = Dense(units=1, activation="linear") (camada_oculta2)

regressor = Model(inputs= camada_entrada, outputs = camada_saida)

regressor.compile(optimizer = 'adam',loss='mse')
regressor.fit(previsores, vendas_global_sales, epochs = 1000, batch_size = 10)


previsao_vendas_global = regressor.predict(previsores)




















