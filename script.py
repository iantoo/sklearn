import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X = load_digits().data
y = load_digits().target

X_train, X_test, y_train, y_test = train_test_split(X, y)
mod = KNeighborsRegressor()
mod.fit(X_train, y_train)

pred = mod.predict(X_test)

plt.scatter(pred, y_test)