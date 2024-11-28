import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression

X = fetch_california_housing().data
y = fetch_california_housing().target

X_train, X_test, y_train, y_test = train_test_split(X, y)

#knneighbors model
mod = KNeighborsRegressor()
#mod.get_params()

modn = GridSearchCV(
    estimator=mod,
    param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
)
modn.fit(X_train, y_train)
pd.DataFrame(modn.cv_results_)

pred = mod.predict(X_test)

print("r2 score: ", r2_score(y_test, pred))

#linear regression model

mod2 = LinearRegression()

mod2.fit(X_train, y_train)
pred2 = mod2.predict(X_test)

print("r2 score: ", r2_score(y_test, pred2))

