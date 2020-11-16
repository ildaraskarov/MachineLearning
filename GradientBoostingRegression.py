from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)

print(X.shape)
print(X.head())
print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = GradientBoostingRegressor(
    max_depth=3,
    n_estimators=33,
    learning_rate=0.1
)
regressor.fit(X_train, y_train)

errors = [mean_squared_error(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)]

best_n_trees = np.argmin(errors)
print("Best number of trees is: {}".format(best_n_trees))

#  construct a model using the best number of trees
best_regressor = GradientBoostingRegressor(
    max_depth=3,
    n_estimators=best_n_trees,
    learning_rate=0.1
)
best_regressor.fit(X_train, y_train)

y_pred = best_regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error is: {}".format(mae))