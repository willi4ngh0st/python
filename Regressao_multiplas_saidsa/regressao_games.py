import pandas as pd
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('games.csv')
colummns = base.columns.tolist()

base = base.drop('Global_Sales', axis=1)
base = base.drop('Other_Sales', axis=1)
base = base.drop('Developer', axis=1)


base = base.dropna(axis = 0)

base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name

base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
vendas_NA = base.iloc[:,4].values
vendas_EU = base.iloc[:,5].values
vendas_JP = base.iloc[:,6].values

labelencoder = LabelEncoder()

previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])

# S 1 0
# R 0 1

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,))

camada_oculta1 = Dense(units = 32, activation = 'sigmoid') (camada_entrada)
camada_oculta2 = Dense(units = 32, activation = 'sigmoid') (camada_oculta1)

camada_saida1 = Dense(units = 1, activation = 'linear') (camada_oculta2)
camada_saida2 = Dense(units = 1, activation = 'linear') (camada_oculta2)
camada_saida3 = Dense(units = 1, activation = 'linear') (camada_oculta2)

regressor = Model(inputs = camada_entrada,
outputs = [camada_saida1, camada_saida2, camada_saida3]
)

regressor.compile(optimizer = 'adam',
loss='mse'
)

regressor.fit(previsores, [vendas_NA, vendas_EU, vendas_JP],
epochs = 5000, batch_size = 100
)
previsao_NA, previsao_EU, previsao_JP = regressor.predict(previsores)
