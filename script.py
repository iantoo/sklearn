import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = fetch_california_housing().data
y = fetch_california_housing().target

X_train, X_test, y_train, y_test = train_test_split(X, y)

#knneighbors model
mod = KNeighborsRegressor()
mod.fit(X_train, y_train)

pred = mod.predict(X_test)

print("r2 score: ", r2_score(y_test, pred))

#linear regression model

mod2 = LinearRegression()

mod2.fit(X_train, y_train)
pred2 = mod2.predict(X_test)

print("r2 score: ", r2_score(y_test, pred2))
