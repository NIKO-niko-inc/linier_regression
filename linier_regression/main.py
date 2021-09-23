import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import mglearn
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# mglearn.plots.plot_linear_regression_wave()

X, y = mglearn.datasets.load_extended_boston()
leb = Ridge
X_train, X_test, y_train, y_test = train_test_split( X, y, random_state=0)
ridge = Ridge(alpha=.1).fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)


print("Правильность на обучающем наборе: {:.2f}".format(ridge.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(ridge.score(X_test, y_test)))


plt.plot(ridge.coef_, 's', label="Гребневая регрессия alpha=1")
plt.plot(ridge.coef_, '^', label="Гребневая регрессия alpha=10")
plt.plot(ridge.coef_, 'v', label="Гребневая регрессия alpha=0.1")
plt.plot(lr.coef_, 'o', label="Линейная регрессия")
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()



