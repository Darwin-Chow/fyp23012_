import lightgbm as lgb
from sklearn.utils import shuffle

# plot graph & XAI
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from plot_graph_tools import plotDataDistribution

# dataFrame package
import pandas as pd
import numpy as np
from numpy import average

# pre-processing
from sklearn.preprocessing import normalize, scale, LabelEncoder, StandardScaler


# result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error

import traceback 


# save model
import pickle

# import function from another files
from model_setup import setupData
from model_train_test import evaluateModel, scoringObj


def main():
    
    # -------------- DEBUG -------------- #
    # read data
    addrData = pd.read_csv("combined_fix_bug.csv")
    addrData = addrData.fillna(0)
    addrData = shuffle(addrData)

    # Select Features
    # 35 Features
    X = addrData[['balance', 'stdev_gas_price', 'mean_transaction_fee', 'max_transaction_fee', 'min_transaction_fee', 'stdev_transaction_price', 'uniq_send_address_num', 'uniq_receive_address_num', 'mean_gas_price', 'time_diff_between_max_balance_and_last_tx', 'mean_time_between_recv_tx', 'highestBalance', 'lowestBalance', 'internal_min_val_recv', 'time_diff_between_min_balance_and_first_tx', 'time_diff_between_max_balance_and_first_tx', 'time_diff_between_min_balance_and_last_tx', 'mean_time_between_send_tx', 'min_gas_price', 'mean_time_between_tx', 'stdev_val_recv', 'transactionCount', 'sendAmount', 'tokenAmount', 'max_gas_price', 'num_of_normal_transaction', 'out_transaction_percent', 'in_transaction_percent', 'time_between_first_and_last_tx', 'min_val_recv', 'stdev_val_send', 'max_val_send', 'min_val_send', 'mean_val_send', 'mean_val_recv']]
    y = addrData['flag']
    
    # Get Test datapoint
    test_data = X.head(5)
    test_ans = y.head(5)
    
    # Standardize the data
    with open('./scaler.pkl','rb') as f:
        sc = pickle.load(f)

    test_data_scaled = sc.transform(test_data)
    test_data_scaled = pd.DataFrame(test_data_scaled, columns = test_data.columns)

    X_scaled = sc.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
    # -------------- DEBUG -------------- #
    
    
    # X, y, X_final_test, y_final_test, features_names = setupData()
    
    
    
    # Print out the test data
    print("\n--------------------\n")
    # print("*** Data to be Tested:\n", test_data_scaled)
    
    
    # Load Model from file
    print("\n*** Loading Model ***\n")
    lgbm_model = lgb.Booster(model_file='lgbm_eth_model.txt')
    
    probas = lgbm_model.predict(test_data_scaled)
    probas_dec = list(map(lambda x: float(f'{x * 100 :2.4f}'), probas))
    
    pred = (probas > 0.5).astype("int")
    
    
    y_pred = lgbm_model.predict(X_scaled)
    y_pred = (y_pred > 0.5).astype("int")
    
    
    print("\n*** Predict Result ***\n")
    print("Prob (1):\t", probas_dec)
    print("Predict:\t", pred)
    print("Answer:\t", test_ans.to_numpy())
    
    # print("\n *** All Data Summary ***\n", classification_report(y_true=y, y_pred= y_pred, target_names=['normal class', 'illicit class'], digits=4))

    
    # print(scorings_name + "\n")
    
    # evaluateModel(model=lgbm_model, name="LightGBM", X=X_scaled, y=y, cv_num=10, scoring_list=scoring_list)
    
    
    print("\n--------------------\n")
    
    
    
    return




if __name__ == "__main__":
    main()


