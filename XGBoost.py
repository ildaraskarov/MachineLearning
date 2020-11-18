import pandas as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import os
import glob
import pathlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from Telco_customer_churn_data import preprocess_Telco_customer_churn_data


filename = 'data/Telco_customer_churn.csv'
X_train, X_test, y_train, y_test = preprocess_Telco_customer_churn_data(filename)

param_grid = {
    'max_length': [3, 4, 5],
    'learning_rate': [0.1, 0.2, 0.5, 1],
    'gamma': [0, 0.25, 0.5, 1],
    'reg_gamma': [0, 5.0, 10.0, 15.0, 20.0 ],
    'scale_pos_weight': [1, 3, 5]
}


optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective= 'binary:logistic',
                                subsample= 0.9,
                                colsample_bytree= 0.5),
    param_grid= param_grid,
    scoring= 'roc_auc',
    verbose= 0,
    n_jobs=4,
    cv= 3
)

optimal_params.fit(
    X_train,
    y_train,
    early_stopping_rounds= 15,
    eval_metric= 'auc',
    eval_set= [(X_test, y_test)],
    verbose= False
)

print("===================== Best  Parameters =================================")
print(optimal_params.best_params_)

#Running XGBoost with the bset parameters

clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            gamma=optimal_params.best_params_['gamma'],
                            learning_rate= optimal_params.best_params_['learning_rate'],
                            max_depth= optimal_params.best_params_['max_length'],
                            reg_lambda= optimal_params.best_params_['reg_gamma'],
                            scale_pos_weight= optimal_params.best_params_['scale_pos_weight'],
                            subsample= 0.9,
                            colsample_bytree= 0.5)


clf_xgb.fit(X_train, y_train, verbose=True,
            early_stopping_rounds=15,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

print("=======================================")
plot_confusion_matrix(clf_xgb, X_test, y_test, values_format='d',
                      display_labels=['Did not leave', 'Left'])

plt.show()







