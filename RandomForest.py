import pandas as pd
import numpy as np
import os
import glob
import pathlib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from Telco_customer_churn_data import preprocess_Telco_customer_churn_data

filename = 'data/Telco_customer_churn.csv'
X_train, X_test, y_train, y_test = preprocess_Telco_customer_churn_data(filename)
#print(X_train.shape)


param_grid = {
    'bootstrap': [True],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [200, 400, 700, 1000]
}

rf = RandomForestClassifier(random_state= 42)

optimal_params = GridSearchCV(estimator= rf,
                      param_grid= param_grid,
                      cv = 5,
                      )

optimal_params.fit(X_train, y_train)

print("==========  Best Parameters of the model from Grid Search  =============")
print(optimal_params.best_params_)



rf_best = RandomForestClassifier(random_state=42,
                                 max_features= optimal_params.best_params_['max_features'],
                                 n_estimators= optimal_params.best_params_['n_estimators'],
                                 max_depth= optimal_params.best_params_['max_depth'],
                                 min_samples_leaf= optimal_params.best_params_['min_samples_leaf'])

rf_best.fit(X_train, y_train)

prediction = rf_best.predict(X_test)
print("=======================================")
print("ROC AUC SCORE for Random Forest on CV data: ", roc_auc_score(y_test,prediction))

print("=======================================")
plot_confusion_matrix(rf_best, X_test, y_test, values_format='d',
                      display_labels=['Did not leave', 'Left'])

plt.show()