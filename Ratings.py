# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the df
df = pd.read_csv('googleplaystore-old.csv')

#Rearranging data
df = df[['App', 'Category','Size','Type','Price','Content Rating', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver', 'Rating', 'Reviews' ,'Installs']]
nan_df = df[df.isna().any(axis=1)]
df = df.dropna(how='any')
df = df[df.Size != 'Varies with device']
df_temp = df
df_temp.Size = df_temp.Size.str[0:-1]
df_temp.Size = df_temp.Size.astype('float')

df_temp.Price = df_temp.Price.str.replace('$', '')
df_temp.Price = df.Price.astype('float')

X_temp = df.iloc[:, 1:7].values
y = df.Rating.astype('float').values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('ohe',OneHotEncoder(),[0,2,4,5])],remainder='passthrough')
X = ct.fit_transform(X)
X = X.toarray()

#Onehotencoder
'''ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[['Genres']])
ohe.fit_transform(df[['Category'],['Type'],['Genres'],['Content Rating']])'''

# Avoiding the Dummy Variable Trap
'''X = np.delete(X,[33],axis=1)'''


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

import statsmodels.api as sm
X = np.append(arr=np.ones((7723,1)).astype(int), values=X,axis=1)
X_opt = X[:,:]
regressor_OLS = sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.04
X_opt = X[:,:]
X_Modeled = backwardElimination(X_opt, SL)

# Splitting the df into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
import math
rmse = math.sqrt(mean_squared_error(y_test,y_pred))
rmse

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

