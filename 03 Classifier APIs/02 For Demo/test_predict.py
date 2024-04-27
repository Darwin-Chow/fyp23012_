import lightgbm as lgb
from sklearn.utils import shuffle

# plot graph & XAI
# import matplotlib.pyplot as plt
# import seaborn as sns

# dataFrame package
import pandas as pd
import numpy as np
from numpy import average

# pre-processing
from sklearn.preprocessing import normalize, scale, LabelEncoder, StandardScaler

import sys
# sys.path.append('../04 Saved Models')

# result
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error

# import traceback 


# save model
import pickle

model_path = "."

def main():
    return



def classifyAccount(addrDict: {}) -> (float):
    
    # -------------- DEBUG -------------- #
    # read data
    # addrData = pd.read_csv("combined_fix_bug.csv")
    # addrData = addrData.fillna(0)
    # addrData = shuffle(addrData)
    

    for k in list(addrDict.keys()):
        addrDict[k] = [addrDict[k]]
    
    
    
    addrData = pd.DataFrame.from_dict(addrDict)
    
    addrData = addrData.fillna(0)
    
    print("\n--- test ---\n", addrData, "\n")
    
    # all features
    # X = addrData.drop(columns=['address', 'token', 
    #                          'firstTransactionTime', 'lastTransactionTime',
    #                         #  'highestBalanceDate', 'lowestBalanceDate', 
    #                         #  'mean_gas_price'
    #                         'time_between_first_and_last_tx',
                            
    #                         'balanceSymbol', 'bandwidth', 'chainFullName', 'chainShortName', 'contractAddress',
    #                         'createContractAddress', 'createContractTransactionHash', 'energy', 'isAaAddress', 'unclaimedVotingRewards',
    #                         'verifying', 'votingRights'
    #                          ])
    
    # ALL features
    X = addrData[['balance', 'transactionCount', 'sendAmount', 'receiveAmount', 'tokenAmount', 'totalTokenValue', 'total_transaction_count', 'num_of_normal_transaction', 'out_transaction_percent', 'in_transaction_percent', 'max_val_send', 'min_val_send', 'mean_val_send', 'stdev_val_send', 'max_val_recv', 'min_val_recv', 'mean_val_recv', 'stdev_val_recv', 'max_gas_price', 'min_gas_price', 'mean_gas_price', 'stdev_gas_price', 'mean_transaction_fee', 'max_transaction_fee', 'min_transaction_fee', 'stdev_transaction_price', 'uniq_send_address_num', 'uniq_receive_address_num', 'zero_val_tx_num', 'zero_val_send_tx_num', 'zero_val_recv_tx_num', 'mean_time_between_tx', 'mean_time_between_send_tx', 'mean_time_between_recv_tx', 'highestBalance', 'lowestBalance', 'num_of_internal_transaction', 'internal_out_transaction_percent', 'internal_in_transaction_percent', 'internal_max_val_send', 'internal_min_val_send', 'internal_mean_val_send', 'internal_stdev_val_send', 'internal_max_val_recv', 'internal_min_val_recv', 'internal_mean_val_recv', 'internal_stdev_val_recv', 'internal_max_gas', 'internal_min_gas', 'internal_mean_gas', 'internal_stdev_gas_price', 'internal_uniq_send_address_num', 'internal_uniq_receive_address_num', 'internal_zero_val_tx_num', 'internal_zero_val_send_tx_num', 'internal_zero_val_recv_tx_num', 'internal_mean_time_between_tx', 'internal_mean_time_between_send_tx', 'internal_mean_time_between_recv_tx', 'time_diff_between_min_balance_and_first_tx', 'time_diff_between_max_balance_and_first_tx', 'time_diff_between_min_balance_and_last_tx', 'time_diff_between_max_balance_and_last_tx']]
    

    # Select Features
    # 35 Features
    # X = addrData[['balance', 'stdev_gas_price', 'mean_transaction_fee', 'max_transaction_fee', 'min_transaction_fee', 'stdev_transaction_price', 'uniq_send_address_num', 'uniq_receive_address_num', 'mean_gas_price', 
                
    #             # 
    #               'time_diff_between_max_balance_and_last_tx', 
                
    #               'mean_time_between_recv_tx', 
    #             # 
    #               'highestBalance', 'lowestBalance', 
                
    #               'internal_min_val_recv', 
    #             # 
    #               'time_diff_between_min_balance_and_first_tx', 
    #               'time_diff_between_max_balance_and_first_tx', 
    #               'time_diff_between_min_balance_and_last_tx', 
                  
    #               'mean_time_between_send_tx', 'min_gas_price', 'mean_time_between_tx', 'stdev_val_recv', 'transactionCount', 'sendAmount', 'tokenAmount', 'max_gas_price', 'num_of_normal_transaction', 'out_transaction_percent', 'in_transaction_percent', 
    #               'time_between_first_and_last_tx', 
    #               'min_val_recv', 'stdev_val_send', 'max_val_send', 'min_val_send', 'mean_val_send', 'mean_val_recv']]
    
    
    # Standardize the data
    with open(f'{model_path}/scaler.pkl','rb') as f:
        sc = pickle.load(f)

    X_scaled = sc.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
    # -------------- DEBUG -------------- #
    
    # Print out the test data
    print("\n--------------------\n")
    # print("*** Data to be Tested:\n", test_data_scaled)
    
    
    # Load Model from file
    print("\n*** Loading Model ***\n")
 
    # with open(f'{model_path}/lgbm_eth_model_allF.txt','rb') as f:
    #     lgbm_model = lgb.Booster(model_str = f.read())
    
    lgbm_model = lgb.Booster(model_file=f'{model_path}/lgbm_eth_model_allF.txt')
    
    probas = lgbm_model.predict(X_scaled)
    probas_dec = list(map(lambda x: float(f'{x * 100 :2.4f}'), probas))
    pred = (probas > 0.5).astype("int")
    
    
    print("\n*** Predict Result ***\n")
    print("Prob (1):\t", probas_dec)
    print("Predict:\t", pred)
    
    
    print("\n--------------------\n")
    
    return (probas_dec)




if __name__ == "__main__":
    main()


