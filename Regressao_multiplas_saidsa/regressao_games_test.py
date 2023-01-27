import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('games.csv')

#save name collumn to compare the price later
nome_jogos = base.Name


#Removing columns
base = base.drop(labels='Name', axis=1)
base = base.drop(labels='Developer', axis=1)
base = base.drop(labels='NA_Sales', axis=1)
base = base.drop(labels='EU_Sales', axis=1)
base = base.drop(labels='JP_Sales', axis=1)
base = base.drop(labels='Other_Sales', axis=1)

base = base.dropna(axis=0)

# working with values that are minors than 1
base = base.loc[base['Global_Sales'] < 1 ]
base = base.loc[base['Global_Sales'] > 0.5 ]

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

#creating neural networking

def criarRede():
    
    camada_entrada = Input(shape=(118,))

    #ativacao = Activation(activation = 'sigmoid')

    camada_oculta1 = Dense(units=59, activation= 'sigmoid') (camada_entrada)
    camada_oculta2 = Dense(units=59, activation= 'sigmoid') (camada_oculta1)

    camada_saida = Dense(units=1, activation="linear") (camada_oculta2)

    regressor = Model(inputs= camada_entrada, outputs = camada_saida)

    regressor.compile(optimizer = 'sgd',loss='mse')
    
    return regressor
    
#activation = sigmoid, kernel_initializer = normal, loos = mse, neurons = 59, optimizer = sgd

regressor =  criarRede()

regressor.fit(previsores, vendas_global, epochs = 5000, batch_size = 10)
previsao_vendas_global = regressor.predict(previsores)
