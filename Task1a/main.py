

import numpy as np

import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv('train.csv', usecols=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13'])
y_train = pd.read_csv('train.csv', usecols=['y'])
X = X_train.to_numpy()
y = y_train.to_numpy()

results=[]

model1 = Ridge(alpha=0.1)
cv1 = KFold(n_splits=10)
scores = cross_val_score(model1,X,y,scoring='neg_root_mean_squared_error',cv=cv1,n_jobs=-1)
scores = np.absolute(np.mean(scores))
results.append(scores)

model2 = Ridge(alpha=1)
cv2 = KFold(n_splits=10)
scores = cross_val_score(model2,X,y,scoring='neg_root_mean_squared_error',cv=cv2,n_jobs=-1)

scores = np.absolute(np.mean(scores))
results.append(scores)

model3 = Ridge(alpha=10)
cv3 = KFold(n_splits=10)
scores = cross_val_score(model3,X,y,scoring='neg_root_mean_squared_error',cv=cv3,n_jobs=-1)

scores = np.absolute(np.mean(scores))
results.append(scores)

model4 = Ridge(alpha=100)
cv4 = KFold(n_splits=10)
scores = cross_val_score(model4,X,y,scoring='neg_root_mean_squared_error',cv=cv4,n_jobs=-1)

scores = np.absolute(np.mean(scores))
results.append(scores)

model5 = Ridge(alpha=200)
cv5 = KFold(n_splits=10)
scores = cross_val_score(model5,X,y,scoring='neg_root_mean_squared_error',cv=cv5,n_jobs=-1)

scores = np.absolute(np.mean(scores))
results.append(scores)

results = np.transpose(results)

dOut = pd.DataFrame(results)
dOut.to_csv('results.csv',index=False,header=False)









