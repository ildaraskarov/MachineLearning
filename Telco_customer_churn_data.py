import pandas as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import pathlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def preprocess_Telco_customer_churn_data(filename):

    core_path = os.getcwd()
    path_to_file = os.path.join(core_path, filename)

    df = pd.read_csv(path_to_file)
    df.drop('customerID', axis=1, inplace=True)
    df.loc[(df['TotalCharges'] == ' '), 'TotalCharges'] = 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df.replace(' ', '_', regex=True, inplace=True)
    X = df.drop('Churn', axis=1).copy()
    y = df['Churn'].copy()
    X_encoded = pd.get_dummies(X, columns= ['gender',
                                        'SeniorCitizen',
                                        'Partner',
                                        'Dependents',
                                        'PhoneService',
                                        'MultipleLines',
                                        'InternetService',
                                        'OnlineSecurity',
                                        'OnlineBackup',
                                        'DeviceProtection',
                                        'TechSupport',
                                        'StreamingTV',
                                        'StreamingMovies',
                                        'Contract',
                                        'PaperlessBilling',
                                        'PaymentMethod'])

    y = y.replace({'No': 0, 'Yes': 1})

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, stratify=y)

    return X_train, X_test, y_train, y_test