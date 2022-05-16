from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X = pd.read_csv('train.csv', usecols=['x1', 'x2', 'x3', 'x4', 'x5'])
y = pd.read_csv("train.csv",usecols=['y'])
X=X.to_numpy()



X_transform = np.concatenate((X,np.power(X,2),np.exp(X),np.cos(X),np.ones((np.shape(X)[0],1))),axis=1)
reg = LinearRegression().fit(X_transform,y)
coef = reg.coef_[0,:20]
p0 = reg.intercept_
out = np.concatenate((coef,p0),axis=0)

dOut = pd.DataFrame(out)
dOut.to_csv('result.csv',index=False,header=False)










