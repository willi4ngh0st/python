from keras.layers import Dense, Dropout, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#importing the libraries
import pandas as pd

cars = pd.read_csv('CarPrice_Assignment.csv')
cars.head()
cars.describe()

columns = cars.columns.to_frame()

cars.drop(['car_ID'],axis=1,inplace=True)
cars.drop(['symboling'],axis=1,inplace=True)
cars.drop(['fueltype'],axis=1,inplace=True)
cars.drop(['enginelocation'],axis=1,inplace=True)


#Splitting company name from CarName column
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['CarName'],axis=1,inplace=True)

Car_brands = cars.CompanyName.unique()

cars.CompanyName = cars.CompanyName.str.lower()

def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)
    
replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

Car_brands = cars.CompanyName.unique()

preco = cars.price.unique()

previsores = cars.iloc[:,0:21].values


# teste

def encode_column(column):
    if isinstance(column[0], str):
        return labelencoder.fit_transform(column)
    else:
        return column

for index in range(previsores.shape[1]):
    previsores[:, index] = encode_column(previsores[:, index])

#end teste

labelencoder = LabelEncoder()

"""

worked but wrong

for index in range(previsores.shape[1]):
    inchk = index
    obj = previsores[0:1,inchk]
    valuechk = list(obj)
    valuechk = valuechk[0]
    if isinstance(valuechk, str):
        previsores[:,index] = labelencoder.fit_transform(previsores[:,index])
    else:
        print()

"""




cars.loc[cars.duplicated()]
cars.columns


arr = [1, 2, 3, 4, 5]
var = arr[2]
print(type(var))

arr = ["apple", "banana", "cherry"]
var = arr[1]
print(type(var))
