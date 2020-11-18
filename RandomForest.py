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

from XGBoost import preprocess_Telco_customer_churn_data

filename = 'data/Telco_customer_churn.csv'
X_train, X_test, y_train, y_test = preprocess_Telco_customer_churn_data(filename)
print(X_train.shape)

param_grid = {
    'bootstrap': [True],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [3, 4, 5],
    'n_estimators': [200, 400, 700, 1000]
}

#rf = RandomForestClassifier(random_state= 42)

#CV_rfc = GridSearchCV(estimator= rf,
#                      param_grid= param_grid,
#                      cv = 5,
#                      )

#CV_rfc.fit(X_train)