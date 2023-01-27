import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
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



def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    
    regressor = Sequential()
    regressor.add(Dense( units = neurons, activation = activation,kernel_initializer = kernel_initializer, input_dim = 118 ))
    regressor.add(Dense( units = neurons, activation = activation, kernel_initializer = kernel_initializer ))
    regressor.add(Dense( units = neurons, activation = activation, kernel_initializer = kernel_initializer ))
    regressor.add(Dense( units=1, activation='sigmoid' ))
    regressor.compile(optimizer = optimizer,loss=loos)
    
    return regressor
    
regressor = KerasRegressor(build_fn=criarRede,
                           epochs = 100,
                           batch_size = 10
                           )

parametros = {
              'optimizer': ['adam','sgd'],
              'loos': ['mse','mae'],
              'kernel_initializer': ['normal'],
              'activation': ['sigmoid','relu'],
              'neurons': [59]}
              
grid_search = GridSearchCV(estimator = regressor, 
                           param_grid = parametros,
                           cv = 5
                           )

#grid_search = grid_search.fit(previsores, vendas_global)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
