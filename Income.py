import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math

def data_smoothening(df, column1, column2, m):
    mean = df[column2].mean()
    agg = df.groupby(column1)[column2].agg(['count', 'mean'])
    count = agg['count']
    mean = agg['mean']
    
    smooth = (count * mean + m * mean) / (count + m)
    
    return df[column1].map(smooth)

df_train = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
df_test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
df_train  = df_train.rename(index=str, columns={"Income in EUR" : "Income"})


df_concat = pd.concat([df_train, df_test], sort=False)

df_concat = df_concat.drop("Instance", axis=1)     
df_concat = df_concat.rename(index=str, columns={"Income in EUR" : "Income"})

df_concat['Gender'] = df_concat['Gender'].replace('0', "other") 
df_concat['Gender'] = df_concat['Gender'].replace('unknown', pd.np.nan) 

df_concat['UniversityDegree'] = pd.factorize(df_concat['University Degree'])[0]
df_concat = df_concat.drop('University Degree', axis = 1)

df_concat['Hair Color'] = df_concat['Hair Color'].replace(['0', 'Unknown'], pd.np.nan) 

df_concat['Country'] = data_smoothening(df_concat, 'Country', 'Income', 2)
df_concat['Profession'] = data_smoothening(df_concat, 'Profession', 'Income', 50)


df_concat.drop("Year of Record", axis=1)
df_concat.drop("Country", axis=1)
df_concat.drop("Wears Glasses", axis=1)

data = pd.get_dummies(df_concat, columns=["Gender"], drop_first = True)
data = pd.get_dummies(data, columns=["Hair Color"], drop_first = True)
pd.set_option('display.max_columns', 100)

X = data[0:len(df_train)]


X["Year of Record"].fillna((X["Year of Record"].mean()), inplace=True )
X["Age"].fillna((X["Age"].mean()), inplace=True )
X["Profession"].fillna((X["Profession"].mean()), inplace=True )

X.isnull().sum()
Y = X[["Income"]]
X = X.drop("Income", axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=100)
RFR = RandomForestRegressor(n_estimators= 900, random_state=100)
model = RFR.fit(X_train, Y_train)

ypred = model.predict(X_test)
mse = mean_squared_error(Y_test, ypred)
rmse = math.sqrt(mse)
print(rmse)
X_test = data[len(df_train):]
X_test = X_test.drop("Income", axis=1)

X_test["Year of Record"].fillna((X_test["Year of Record"].mean()), inplace=True )
X_test["Age"].fillna((X_test["Age"].mean()), inplace=True )
X_test["Profession"].fillna((X_test["Profession"].mean()), inplace=True )
X_test["Country"].fillna((X_test["Country"].mean()), inplace=True )
Y_pred = model.predict(X_test)
Y_pred = pd.DataFrame(Y_pred)

np.savetxt("Income.csv", Y_pred, delimiter=",", fmt='%s', header="Income")
