
# dataFrame package
import pandas as pd
import numpy as np
from numpy import average

import traceback 

# sklearn tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import normalize, scale, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error, make_scorer

# pre-processing
from sklearn.decomposition import PCA
from sklearn.utils import resample


# import all models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from hyperopt import hp


from xgboost import XGBClassifier
import lightgbm as lgb

# plot graph & XAI
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from plot_graph_tools import plotDataDistribution, plotROC_curve, plotROC_curve_pred


# import function from another files
from model_setup import setupData




# chekcing progress// DEBUG
from tqdm import tqdm


# init
labelEncoder = LabelEncoder()


class scoringObj:
  
  def __init__(self, scoring_metrics: str, name: str, evaluate_metrics):
    self.scoring_metrics = scoring_metrics
    self.name = name
    self.evaluate_metrics = evaluate_metrics



# 
# done 1) handle symbol 「"」error in balance (hi/low)
# done 2) strign mapping error
# done 3) values too large error
# 
# 

def main():
  
  print("\n** Setting up Data...\n")
  X, y, X_final_test, y_final_test = setupData()
  
  
  print("\n** Building the Model...\n")
  trainModel(X, y, X_final_test, y_final_test)
 
 


# -----------------------------

def trainModel(X, y, X_test, y_test):
  
  data = (X, y, X_test, y_test)
  features_names = X.columns.values.tolist()
  
  
  
  # return
  
  # cut the data into test & train data
  # X = PCA(n_components=2).fit_transform(X)
  
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33, stratify=y)


  # init models
  logiR = LogisticRegression(solver='lbfgs', max_iter=1000)
  rfc = RandomForestClassifier(random_state=42)
  knnc = KNeighborsClassifier(n_neighbors=11, p = 2, weights='distance', algorithm='auto')
  xgbc = XGBClassifier(random_state=42)
  lgbmc = lgb.LGBMClassifier(is_unbalance=True, 
                             verbose = 0, 
                             learning_rate = 0.5, 
                             n_estimators = 200, 
                             num_leaves = 19,
                             min_child_samples = 120,
                             
                            #  old
                            #  learning_rate= 0.2, 
                            #  n_estimators= 140, 
                            #  num_leaves= 19,
                            
                            #  feature_fraction= 0.8,
                            #  max_depth = 7,
                            #  min_child_sample= 20,
                            #  bagging_freq=2,
                             
                             random_state=42)
  # lgbmc = lgb.LGBMClassifier(**lgbm_params, is_unbalance=True, verbose= 0)
  svmc = SVC(gamma=0.7)
  
  
  
  # 定义 Stacking 模型
  estimators = [
    ('svc', svmc),
    ('rfc', rfc),
    ('knn', knnc),
    ('logi', logiR),
    ('dt', DecisionTreeClassifier())
  ]
  
  stack_cf = StackingClassifier(
      estimators=estimators, final_estimator= logiR
  )
  # stacking_model = StackingClassifier(classifiers=[logiR, rfc, svmc], meta_classifier=meta_model)
  
  
  models = [logiR, 
            rfc, 
            knnc, 
            xgbc, 
            lgbmc, 
            svmc,
            stack_cf
            ]
  # models = [logiR, rfc, knnc, xgbc, lgbmc, svmc, stack_cf]
  
  
  
  model_name = ["LogisticReg", 
                "RF", 
                "KNN", 
                "XGBoost", 
                "LightGBM", 
                "SVM",
                "Stack (SVC, RFC, kNN, logiR, dt)"
                ]

  
  
  scorings_di = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'specificity':  make_scorer(recall_score, pos_label=0),
           'f1': make_scorer(f1_score),
           'f1_macro': make_scorer(f1_score, average='macro'),
           'f1_weighted': make_scorer(f1_score, average='weighted')
  }

  scorings_name = "\t".join([x for x in list(scorings_di.keys())])

  
  # ---------------  Evaluation of Different Models --------------------
  print(f"** Model\t{scorings_name}\n")

  
  # ---- Testing CV for various Models -------
  
  cv_num = 10
  
  for model, name in zip(models, model_name):
    # pass
    evaluateModel(model=model, 
                  name=name, 
                  X=X, y=y, 
                  cv_num=cv_num, 
                  scoring_list=scorings_di,
                  
                  )

      
    
  # //-------------  END of Evaluation of Different Models --------------------//
  
  # // ---------- Fine-turning Models -------------
  
  ## 撰寫訓練用的參數
  param_grid = {
      # 'num_leaves': range(5, 32),
      # 'learning_rate': [0.05, 0.1, 0.2, 0.3 ,0.4, 0.5],
      # 'n_estimators': range(10, 150, 5)
      
      # 'num_leaves': [5, 19, 31],
      # 'learning_rate': [0.2, 0.5, 0.8],
      # 'n_estimators': [140, 170, 200],
      
      # # 'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
      # # 'bagging_freq': [2, 4, 6, 8],
      # 'max_depth': [3, 4, 5], # 2^(leaves) > max_depth
      # # 'min_child_sample': [20, 50, 100] --> no such things
      
      
      #subsample代表為採樣數
      #因為'goss'並無區分是否含有採樣數 所以給予1(不採樣)
      # 'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
      #                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
      #                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
      'num_leaves': [5, 19, 31],
      'n_estimators': [140, 170, 200],
      'learning_rate': [0.2, 0.5, 0.8],
      # 'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
      'min_child_samples': range(20, 500, 100),
      # 'reg_alpha': range(0.0, 1.0, 0.1),
      # 'reg_lambda': range(0.0, 1.0, 0.2),
      # 'colsample_bytree': range(0.6, 1.0, 0.1)
  }
  
  lbgm_test = lgb.LGBMClassifier(is_unbalance=True, verbose= 1)
  
  # fineTuningModel(lbgm_test, param_grid, data)
  
  
  # // ------------ Evaluation of Final Model (LightGBM) -------------
  
  print("\n** Results of Best Model (LightGBM) **\n")
  
  # Create LightGBM datasets for training and testing
  lgb_train_data = lgb.Dataset(X, label=y)
  lgb_test_data = lgb.Dataset(X_test, label=y_test, reference=lgb_train_data)
  lgb_params = {
    'is_unbalance': True, 
    'verbose': 0, 
    'learning_rate': 0.2, 
     'n_estimators': 140, 
     'num_leaves':19,
     'random_state':42,
     
  }
  lgb_params['metric'] = ['auc', 'binary_logloss']
  bst = lgb.train(lgb_params, lgb_train_data, 
                  num_boost_round=100, 
                  valid_sets=[lgb_test_data],
                  callbacks=[lgb.early_stopping(stopping_rounds=5)]
                  )
  
  # Make predictions
  y_pred = bst.predict(X_test)
  
# Convert probabilities to binary predictions
  y_pred_binary = (y_pred > 0.5).astype(int)
  
  lgbmc.fit(X=X, y=y)
  y_pred = lgbmc.predict(X=X_test)
  
  
  # TODO: save the model
  # print("### SAVE MODEL... ###")
  # lgbmc.booster_.save_model("lgbm_eth_model_allF.txt")
  # testLoadModel(X=X_test, y=y_test)
  
  # -- Feature Enginnering --
  # feature_ranking_extraction(lgbmc, X, y, X_test, y_test)
  

  print("** lightGBM build-int API ***\n", classification_report(y_true=y_test, y_pred= y_pred_binary, target_names=['normal class', 'illicit class'], digits=4))
  print("** lightGBM sklearn API ***\n", classification_report(y_true=y_test, y_pred= y_pred, target_names=['normal class', 'illicit class'], digits=4))
  
  # plotROC_curve(estimator=lgbmc, title="ROC curve for LightGBM in  training", X_test=X_test, y_test=y_test)
  # plotROC_curve_pred("ROC curve for LightGBM in Testing", y_test=y_test, y_pred=y_pred)
  
  # TEST
  print("\n** Results of Best Model (XGBoost) **\n")
  xgbc.fit(X=X, y=y)
  y_pred_xgb = xgbc.predict(X=X_test)
  print(classification_report(y_true=y_test, y_pred= y_pred_xgb, target_names=['normal class', 'illicit class'], digits=4))
  
  
  # # get Figures for data
  # plotDataDistribution(y_test)
  genConfusionMatrix(y_test=y_test, y_pred=y_pred)


  # ----- FOR XAI techniques -----
  
  XAI_model(model=lgbmc, X=X, max_display=10)


  # ----- For testing SVM -----
  
  # SVM REF: https://medium.com/@search.psop/python學習筆記-16-機器學習之svm實作篇-719d98c68ce3
  # --> fine-turing models
  
  return




def fineTuningModel(model, param_grid, data):
  
  # 
  print("\n## Fine Tuning Model... ###\n")
  
  # Initialize an empty dictionary to store the best hyperparameters and their values
  best_hyperparameters = {}
  best_values = {}
  
  X_train, y_train, X_test, y_test = data
  
  # Initialize the LightGBM classifier
  
  specificity = make_scorer(recall_score, pos_label=0)
  
  
  # Initialize GridSearchCV for hyperparameters
  grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                            scoring=specificity, cv=10, verbose= 2)
  
  # Fit the model to the training data to search for the best hyperparameters
  grid_search.fit(X_train, y_train)
  
  # Get the best hyperparameters and their values
  best_params = grid_search.best_params_
  best_hyperparameters = list(best_params.keys())
  best_values = list(best_params.values())
  
  print("Best_param:\n", best_params)
  
  # Train a LightGBM model with the best hyperparameters
  best_model = lgb.LGBMClassifier(**best_params)
  best_model.fit(X_train, y_train)
  
  # Make predictions on the test set using the best model
  y_pred = best_model.predict(X_test)
  
  print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
  
  # //----------------------------


def evaluateModel(model, name, X, y, cv_num, scoring_list, isPipe = False):
  
  # scorings = [x.scoring_metrics for x in scoring_list]
  # evaluate_metrics = [x.evaluate_metrics for x in scoring_list]
  
  # Define multiple metrics
  scorings = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'specificity':  make_scorer(recall_score, pos_label=0),
           'f1': make_scorer(f1_score),
           'f1_macro': make_scorer(f1_score, average='macro'),
           'f1_weighted': make_scorer(f1_score, average='weighted')
          }
  
  scoring_names = [s for s in list(scorings.keys())]
  
  
  if (not isPipe):
    
    try:
        # scores = cross_validate(model, X=X, y=y, cv= cv_num, scoring=scorings, n_jobs=4)
        # scores_value = [float(scores[s].mean()) for s in evaluate_metrics]
        
        scores = cross_validate(model, X=X, y=y, cv= cv_num, scoring=scorings, n_jobs=4)
        scores_value = [float(scores[f'test_{s}'].mean()) for s in scoring_names]
        
        # print(scores)
        
        print(getEvaluationMetric(modelName = name, scores = scores_value))
        print()
      
    except Exception as e:
        traceback.print_exc()
        print(f"\n!!! ERROR: {e} !!!\n")
  
  
  
  # # 
  # else:
  #   print("\n-- Pipeline Mode --\n")
  #   pipe_md = Pipeline([('scl', StandardScaler()),
  #                      (name, model)]
  #                      )
  #   try:
  #       scores = cross_validate(pipe_md, X=X, y=y, cv= cv_num, scoring=scorings, n_jobs=4)
  #       scores_value = [float(scores[s].mean()) for s in evaluate_metrics]
        
  #       print(getEvaluationMetric(modelName = name, scores = scores_value))
  #       print()
      
  #   except Exception as e:
  #       traceback.print_exc()
  #       print(f"\n!!! ERROR: {e} !!!\n")
    
    
    


def feature_ranking_extraction(model, X, y, X_test, y_test):
  
  model.booster_.feature_importance()

  # TODO: Feature Selection:
  #       importance of each attribute
  fea_imp_ = pd.DataFrame({'cols': X.columns.values.tolist(), 'fea_imp': model.feature_importances_})
  
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False))
  
  # TODO: Recursive Feature Elimination(RFE)

  # create the RFE model and select 10 attributes
  
  # TODO: 
  # 1) test between 35 - 40 
  # 2) find out what is R2-score, AUC, ...
  num_of_features = 35 # 35 > 30 = same accuracy >= 40 >= 25 > 20 >= 27
  rfe = RFE(estimator=model, n_features_to_select=num_of_features)
  rfe = rfe.fit(X, y)
  rfe_pred = rfe.predict(X_test)

  # summarize the selection of the attributes
  print(rfe.support_)
  
  
  # summarize the ranking of the attributes
  fea_rank_ = pd.DataFrame({'cols': X.columns.values.tolist(), 'fea_rank': rfe.ranking_})
  
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    fea_rank_ = fea_rank_.sort_values(by='fea_rank', ascending = True)
    print(fea_rank_.sort_values(by='fea_rank', ascending = True))
    print(fea_rank_['cols'].head(num_of_features).tolist())
  
  

  print(f"** RFE #features={num_of_features} ***\n", classification_report(y_true=y_test, y_pred= rfe_pred, target_names=['normal class', 'illicit class'], digits=4))
  # evaluateModel(model = rfe, name = f"REF #{num_of_features}", X = X, y=y, cv_num=10, scoring_list=scoring_list)
  
  #  create pipeline
  # rfe = RFECV(estimator=lgbmc)
  # model = lgb.LGBMClassifier(is_unbalance=True, verbose= 0, learning_rate= 0.2, n_estimators= 140, num_leaves= 19, random_state=42)
  # pipeline = Pipeline(steps=[('s',rfe),('m',model)])
  # # evaluate model
  # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=33)
  # n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  # # report performance
  # print(f'\n** RFE Accuracy: {np.mean(n_scores): .3f} {np.std(n_scores) : .3f} **\n no. of Features: {rfe.n_features_} **\n' )
  
  # print(fea_rank_)
  
  
  

def testLoadModel(X, y):
  # Load Model from file
    print("\n*** Loading Model ***\n")
    lgbm_model = lgb.Booster(model_file='lgbm_eth_model_allF.txt')
    
    probas = lgbm_model.predict(X)
    pred = (probas > 0.5).astype("int")
    
    print("\n *** Load Data Summary ***\n", classification_report(y_true=y, y_pred= pred, target_names=['normal class', 'illicit class'], digits=4))

# function to get Evaluation Metrics
def getEvaluationMetric(modelName: str, scores: []) -> str:

    
    scores_str = ' '.join(list(map(convertFloatToStr, scores)))
  
    return f"{modelName: >8}\t{scores_str}"


def convertFloatToStr(num: float):
  return f'{num: .4f}'
  

def genConfusionMatrix(y_test,y_pred):
  
    annot_values = [["TN", "FP"], ["FN", "TP"]]
    txt_label = ["Normal", "Illicit"]
  
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8,6))
    # fig, ax = plt.subplots()
    
    
    sns.heatmap(cm, square=True, annot=True, fmt='d', annot_kws={'va':'top', "size": 18}, linecolor='white', cmap='RdBu', linewidths=1.6, cbar=False)
    sns.heatmap(cm, square=True, annot=annot_values, annot_kws={'va':'bottom', "size": 16}, fmt="", linecolor='white', cmap='RdBu', linewidths=1.6, cbar=False)
    
    plt.title("Confusion Matrix for LightGBM in testing", fontsize=22)
    plt.xlabel('Predicted label', fontsize=18)
    plt.ylabel('True label', fontsize=18)
    
    # # Adding legend
    # ax.legend(txt_label,
    #       title ="Types",
    #       loc ="center left",
    #       bbox_to_anchor =(1, 0, 0.5, 1), fontsize=14)
    
    plt.show()



def XAI_model(model, X, max_display):
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
  
  # REF: custom plot summary plot:
  # --> https://medium.com/@wellylin8916/custom-shap-summary-plot-b3167d61ca4d

  # summarize the effects of all the features
  shap.summary_plot(shap_values[1], X, 
                    class_names=['normal', 'illicit'], 
                    max_display=max_display,
                    plot_size = (10.0, 8.0),
                    show = False)
  
  plt.yticks(fontsize=15)
  
  plt.title("Effect of Top 10 important features of illicit account", fontsize=18, pad=15, weight='bold')
  plt.show()
  # shap.plots.waterfall(shap_values)

if __name__ == "__main__":
  main()








# ------ BACKUP -----

  # for train_index, test_index in kf.split(X):
    
  #   break
    
  #   print("-------")
    
  #   X_train, X_test = X[train_index], X[test_index]
  #   y_train, y_test = y[train_index], y[test_index]
    
    
    
  #   # ----  For testing Random Forest -------
  #   rfc = RandomForestClassifier()

  #   rfc.fit(X=X_train, y=y_train)
    
  #   rfc_pred = rfc.predict(X_test)
    
  #   print(getEvaluationMetric("Random Forest", y_test=y_test, y_pred=rfc_pred)) 

  #   # ----- For testing XGBoost ------
  #   xgbc = XGBClassifier()

  #   xgbc.fit(X=X_train, y=y_train)
    
  #   xgbc_pred = xgbc.predict(X_test)
    
  #   print(getEvaluationMetric("XGBoost", y_test=y_test, y_pred=xgbc_pred))    
    
  #   # ----- FOR Testing LightGBM
    
  #   lgbmc = lgb.LGBMClassifier(is_unbalance=True, verbose= 0)
  #   lgbmc.fit(X_train,y_train)  
    
  #   lbgm_pred = lgbmc.predict(X_test)
  #   print(getEvaluationMetric("LightGBM", y_test=y_test, y_pred=lbgm_pred))
    
  #   # ----- FOR Testing KNN
    
  #   knnc=KNeighborsClassifier(n_neighbors=11, p=2, weights='distance', algorithm='auto')
  #   knnc.fit(X_train,y_train)
    
  #   knnc_pred = knnc.predict(X_test)
  #   print(getEvaluationMetric("KNN", y_test=y_test, y_pred=knnc_pred))
    
  #   print("-------")
  
