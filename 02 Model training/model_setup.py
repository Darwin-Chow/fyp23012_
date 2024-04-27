
# tools
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# preprocess & Resample
from sklearn.preprocessing import normalize, scale, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Sampling
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression


from enum import Enum



# save model
import pickle


from plot_graph_tools import plotDataDistribution, plotROC_curve_pred


# Global variables
class SamplingMethods(Enum):
  NONE = 0
  ENN = 1
  SMOTE_ENN = 2
  ADASYN_ENN = 3
  SMOTE_TomekLinks = 4


class FeatureScalingMethods(Enum):
  NONE = 0
  STANDARDIZE = 1
  NORMALIZE = 2



def test():
  setupData()
  
  return




def setupData(isUnsampled = False, isStandardized = True):
    
  addrData = pd.read_csv("combined_fix_bug.csv")
  
  addrData = addrData.fillna(0)
  
  addrData = shuffle(addrData, random_state=33)

  # to_scale = ['sendAmount', 'receiveAmount']
  # addrData.loc[:, to_scale] = scale(addrData[to_scale])

  X = addrData.drop(columns=['index', 'address', 'flag', 'token', 
                             'firstTransactionTime', 'lastTransactionTime',
                             'highestBalanceDate', 'lowestBalanceDate', 
                            #  'mean_gas_price'
                            # 'time_between_first_and_last_tx'
                             ])
  
  # 35 Features - ('time_between_first_and_last_tx')
  # X = addrData[['balance', 
  #               'stdev_gas_price', 
  #               'mean_transaction_fee', 
  #               'max_transaction_fee', 
  #               'min_transaction_fee', 
  #               'stdev_transaction_price', 
  #               'uniq_send_address_num', 
  #               'uniq_receive_address_num', 'mean_gas_price', 'time_diff_between_max_balance_and_last_tx', 'mean_time_between_recv_tx', 'highestBalance', 'lowestBalance', 'internal_min_val_recv', 'time_diff_between_min_balance_and_first_tx', 'time_diff_between_max_balance_and_first_tx', 'time_diff_between_min_balance_and_last_tx', 'mean_time_between_send_tx', 'min_gas_price', 'mean_time_between_tx', 'stdev_val_recv', 'transactionCount', 'sendAmount', 'tokenAmount', 'max_gas_price', 'num_of_normal_transaction', 'out_transaction_percent', 'in_transaction_percent', 
  #               'time_between_first_and_last_tx', 
  #               'min_val_recv', 'stdev_val_send', 'max_val_send', 'min_val_send', 'mean_val_send', 'mean_val_recv']]

  y = addrData['flag']
  
  print(X['receiveAmount'])

  
  
  

  # 你可以輸入你要的特徵清單，dropna丟掉沒意義的值，再計算相關係數corr矩陣
  features_list = list(X.columns.values.tolist())
  corr = addrData[features_list].corr()
  # print(corr)
  
  # print("\n--- Features List ---\n", features_list, "\n")
  
  # 用seaborn heatmap 把相關係數畫出來
  # sns.heatmap(corr, annot = True) # 你可以再加上annot = True，這樣corr value就可以被畫出來。
  # plt.show()

  
  
  # print(X)
  print(y)
  
  
  if (not isStandardized):
    return X, y
  
  features_names = list(X.columns.values.tolist())

  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=33, stratify=y)
  
  # print(list(X_test.columns.values.tolist()))
  
  # 1) Standization of features 
  sc = StandardScaler()
  X_train_scaled = sc.fit_transform(X_train)
  X_test_scaled = sc.transform(X_test)
  
  
  # 2) Normalization of values
  mms = MinMaxScaler()
  X_train_norm = mms.fit_transform(X_train)
  X_test_norm = mms.transform(X_test)
  
  
  
  # save StandarddScaler
  # print("### Saving StandardScaler... ###")
  # with open('./scaler.pkl','wb') as f:
  #   pickle.dump(sc, f)
  
  
  X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
  X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
  
  # 
  X_train_norm = pd.DataFrame(X_train_norm, columns = X_train.columns)
  X_test_norm = pd.DataFrame(X_test_norm, columns = X_test.columns)
  
#   sns.histplot(X_train_scaled['sendAmount'], color = 'blue', bins=50,
# )
#   plt.show()

  # Return Unsampled Datasets
  if (isUnsampled):
    return X, y, X_train, X_train_scaled, y_train, X_test_scaled, y_test
  
  
  #   
  flaged_acc = addrData[addrData["flag"] == 1]
  norm_acc = addrData[addrData["flag"] == 0]
  
  
  # print original
  print("\n*** Data Summary ***\n")
  print("--- Before DownSampling ---")
  print(f"Normal acc: {norm_acc.shape}\nFlag acc: {flaged_acc.shape}")
  
  # test Sampling approaches
  
  
  
  
  # TODO: Test K-fold before standardScaler
  # return X_train, y_train, X_test, y_test
  
  
  
  
  
  # Plot the account distribution before and after the ENN
  # plotDataDistribution(y, "Distribution for types of accounts")
  # plotDataDistribution(y_resample_enn, "Distribution for types of accounts after applying ENN") # --> ONLY train data included
  
  fea_scaling_method = FeatureScalingMethods.STANDARDIZE
  smp_method = SamplingMethods.SMOTE_TomekLinks
  # smp_method = SamplingMethods.NONE
  
  
  
  #  ----- FEATUREs Scaling ------
  print(f"\n### Feature scaling method: {fea_scaling_method.name}  ###\n")
  
  if (fea_scaling_method == FeatureScalingMethods.NONE):
    print("\n!!!! ERROR !!!!\n")
    pass

  elif (fea_scaling_method == FeatureScalingMethods.STANDARDIZE):
    X_train, X_test = X_train_scaled, X_test_scaled

  elif (fea_scaling_method == FeatureScalingMethods.NORMALIZE):
    print("\n!!!! ERROR !!!!\n")
    X_train, X_test = X_train_norm, X_test_norm
  #  ----- ----------  ------



  # evaluate ROC
  lr_samp = LogisticRegression(solver='lbfgs', max_iter=1000)
  rfc_samp = RandomForestClassifier(random_state=42)
  # plot_sampling_graph(lr_samp, X_train, y_train, X_test, y_test, False)
  

  # ----- SAMPLING methods ------
  print(f"\n### Sampling method: {smp_method.name}  ###\n")
  
  if (smp_method == SamplingMethods.NONE):
    pass

  elif (smp_method == SamplingMethods.ENN):
    X_train, y_train = ENN_sampling(X_train=X_train, y_train=y_train)

  elif (smp_method == SamplingMethods.SMOTE_ENN):
    X_train, y_train = SMOTE_and_ENN_sampling(X_train=X_train, y_train=y_train)
    
  elif (smp_method == SamplingMethods.SMOTE_TomekLinks):
    X_train, y_train = SMOTE_TomekLinks_sampling(X_train=X_train, y_train=y_train)

  
  elif (smp_method == SamplingMethods.ADASYN_ENN):
    X_train, y_train = ADASYN_sampling(X_train=X_train, y_train=y_train)

#  ----- ----------  ------



# plot the Sampling Distribution
  # plotDataDistribution(y_train, "Distribution for types of accounts after Sampling (SMOTE + TomekLinks)") # --> ONLY train data included




  return X_train, y_train, X_test, y_test
  # return X_resample_enn, y_resample_enn, X_test, y_test, features_names

# ---- END of MAIN function ----



def inverseScaler(X):
    with open('./scaler.pkl','rb') as f:
      sc = pickle.load(f)
    
    X_inv_scaled = sc.inverse_transform(X)
    
    return X_inv_scaled


def loadScaler():
    with open('./scaler.pkl','rb') as f:
      sc = pickle.load(f)
      
    return sc
  
  
def stdTrainData(X):
  sc = loadScaler()
  X_scaled = sc.transform(X)
  return X_scaled
  

# ----------- Sampling ------------------

def plot_sampling_graph(model, X_train, y_train, X_test, y_test, isProba=False):
  
  # NONE
  # model = LogisticRegression(solver='lbfgs', max_iter=1000)
  model.fit(X_train, y_train)
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
  
  plotROC_curve_pred(title="", method = "none", y_test=y_test, y_pred=y_pred, isShow=False)
  
  
  # ADASYN
  
  # re-define the model will not Affect the results
  # model = LogisticRegression(solver='lbfgs', max_iter=1000)
  X_train_ada, y_train_ada = ADASYN_sampling(X_train=X_train, y_train=y_train)
  model.fit(X_train_ada, y_train_ada)
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
  plotROC_curve_pred(title="", method = "ADASYN", y_test=y_test, y_pred=y_pred, isShow=False)
  
  
  # ENN
  X_train_enn, y_train_enn = ENN_sampling(X_train=X_train, y_train=y_train)
  model.fit(X_train_enn, y_train_enn)
  
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
    
  plotROC_curve_pred(title="", method = "ENN", y_test=y_test, y_pred=y_pred, isShow=False)

  
  # SMOTE + ENN
  X_train_smoteenn, y_train_smoteenn = SMOTE_and_ENN_sampling(X_train=X_train, y_train=y_train)
  model.fit(X_train_smoteenn, y_train_smoteenn)
  
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
    
  plotROC_curve_pred(title="ROC curves for various sampling methods", method = "SMOTE + ENN", y_test=y_test, y_pred=y_pred, isShow=False)
  
  
  
  # ADASYN + ENN
  X_train_adaene, y_train_adaenn = ADASYN_ENN_sampling(X_train=X_train, y_train=y_train)
  model.fit(X_train_adaene, y_train_adaenn)
  
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
    
  plotROC_curve_pred(title="ROC curves for various sampling methods", method = "ADASYN + ENN", y_test=y_test, y_pred=y_pred, isShow=False)
  
  
  # SMOTE + TomekLinks
  X_smoteTome, y_smoteTome = SMOTE_TomekLinks_sampling(X_train=X_train, y_train=y_train)
  
  model.fit(X_smoteTome, y_smoteTome)
  
  if (not isProba):
    y_pred = model.predict(X_test)
  else:
    y_pred = model.predict_proba(X_test)[:, 1]
    
  plotROC_curve_pred(title="ROC curves for various sampling methods", method = "SMOTE + TomekLinks", y_test=y_test, y_pred=y_pred, isShow=True)



def random_sampling(norm_acc, flag_acc):
  norm_acc = resample(norm_acc, 
                      replace=True,
                      n_samples=len(flag_acc),
                      random_state=42
                      )
  
  print("--- Random Down-Sampling ---")
  print(f"Flag acc: {flag_acc.shape}\nNormal acc: {norm_acc.shape}")
  
  
  addrData_downSample = pd.concat([norm_acc, flag_acc])
  
  print(addrData_downSample["flag"].value_counts())
  
  X = addrData_downSample.drop(columns=['index', 'address', 'flag', 'token', 
                             'firstTransactionTime', 'lastTransactionTime',
                             'highestBalanceDate', 'lowestBalanceDate'
                             ])

  y = addrData_downSample['flag']
  
  return X, y


def ENN_sampling(X_train, y_train, log = True) -> ():
  
  # enn = EditedNearestNeighbours(sampling_strategy = 'majority', n_neighbors=11)
  enn = EditedNearestNeighbours(n_neighbors=11)
  
  X_resample_enn, y_resample_enn = enn.fit_resample(X_train, y_train)
  
  if (log):
    print("\n--- After ENN ---\n")
    print(f"tResampled dataset shape: {Counter(y_resample_enn)}")
  
  # Plot the account distribution before and after the ENN
  # plotDataDistribution(y)
  # plotDataDistribution(y_resample_enn) # --> ONLY train data included
  
  return X_resample_enn, y_resample_enn



def SMOTE_and_ENN_sampling(X_train, y_train, log=True) -> ():
  
  # X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
  
  sme = SMOTEENN(random_state=42)
  X_res, y_res = sme.fit_resample(X_train, y_train)
  
  if (log):
    print("\n--- After SMOTE + ENN ---\n")
    print(f'\tResampled dataset shape {Counter(y_res)}\n')
  
  return X_res, y_res



def ADASYN_sampling(X_train, y_train, log=True) -> ():
  ada = ADASYN(random_state = 42)
  
  X_res, y_res = ada.fit_resample(X_train, y_train)
  
  if (log):
    print("\n--- after ADASYN ---\n")
    print(f'\tResampled dataset shape {Counter(y_res)}\n')
  
  return X_res, y_res
  
  
def ADASYN_ENN_sampling(X_train, y_train, log=True) -> ():
  X_ada, y_ada = ADASYN_sampling(X_train, y_train, False)
  X_res, y_res = ENN_sampling(X_ada, y_ada, False)
  
  if (log):
    print("\n--- after ADASYN + ENN ---\n")
    print(f'\tResampled dataset shape {Counter(y_res)}\n')
  
  return X_res, y_res



def SMOTE_TomekLinks_sampling(X_train, y_train, log=True) -> ():
  smote = SMOTE(sampling_strategy='auto', k_neighbors=11, random_state=42)
  X_smote, y_smote = smote.fit_resample(X_train, y_train)
  
  tl = TomekLinks(sampling_strategy="all")
  X_res, y_res = tl.fit_resample(X_smote, y_smote)
  
  
  if (log):
    print("\n--- after SMOTE + TomeLinks ---\n")
    print(f'\tResampled dataset shape {Counter(y_res)}\n')
  
  
  return X_res, y_res
  


# ------------------ Deep learning part ------------------
import itertools

def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example
    
  
  # X['token'] = labelEncoder.fit_transform(X['token'])

  # X = normalize(X)
  # X = X.fillna(X.mean())
  # print(X)
  

if __name__ == "__main__":
  test()
    
    