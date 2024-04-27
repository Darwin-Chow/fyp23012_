

# dataFrame package
import pandas as pd
import numpy as np
from numpy import average
from statistics import mean 

import traceback 

from tqdm import tqdm


import plotly.express as px 

# 
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix

# 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Unsupervised Algorihtms
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture

# 
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# Evaluation Metrics for Unsupervsied
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, mean_squared_error, make_scorer


# plot graph & XAI
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from plot_graph_tools import plotDataDistribution, plotROC_curve, plotROC_curve_pred, plot_3D_graph, plot_gmm, plot_dendrogram

# 
from collections import Counter


# import function from another files
from model_setup import setupData, inverseScaler, stdTrainData


sepChar = "\t"



def main():
    print("\n** Setting up Data...\n")
    # X, y, X_final_test, y_final_test = setupData()
    
    
    X_all, y_all, df_X, X, y, X_test, y_test = setupData(isUnsampled=True)
    
    # X_unStd, y_unStd = setupData(isStandardized=False)
    
    
    # use StratifiedShuffleSplit() 
    strafiedCV(isolationForest, X_all=X_all, y_all=y_all)
    strafiedCV(oneClassSVM, X_all=X_all, y_all=y_all)
    
    # isolationForest(df=df_X, X=X, y=y)
    
    # oneClassSVM(df=df_X, X=X, y=y, X_test=X_test, y_test=y_test)
    
    # -- For inverse Standardization ----
    # X_inv = inverseScaler(X)
    # df = pd.DataFrame(X_inv, columns = X_train_unsamp.columns)
    # df.to_csv(f"X_invScale_data.csv")
    # print(f"\n--Saved to X_invScale_data.csv...\n")
    # ------------------
    
    
    # print("-----\n", X_ori)
    
    # test PCA
    print("\n** Testing PCA,,,\n")
    # PCA_evaluate(X)


    print("\n** Building the Model...\n")
    # trainUnsupervisedModel(X, df_X, y, X_test, y_test)





def oneClassSVM(df, X, y, X_test = [], y_test = []):
    print("----- Testing OneClassSVM -----\n")
    outliers_fraction = 0.3
    # TODO OneClassSVM
    oneSvm = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
    y_pred = oneSvm.fit_predict(X)
    
    mapped_pred = mapPredLabelToLabel(y_pred, 1, -1)
    print(mapped_pred)
    res = retreiveScoreFromY(y=y, y_pred=y_pred)
    return res
    
    # print(classification_report(y_true=y, y_pred= mapped_pred, target_names=['normal class', 'anomaly class'], digits=4))
    
    
    y_test_pred = oneSvm.predict(X_test)
    mapped_pred = mapPredLabelToLabel(y_test_pred, 1, -1)
    print(mapped_pred)
    
    # print(classification_report(y_true=y_test, y_pred= mapped_pred, target_names=['normal class', 'anomaly class'], digits=4))




def isolationForest(df, X, y):
    print("----- Testing isolation Forest -----\n")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=33, stratify=y)
    
    # print(df[['sendAmount', 'receiveAmount']])
    # print("isZero: ", (df['receiveAmount'] == 0).all())
    # print("non-Zero: ", (df[df['receiveAmount'] != 0]['receiveAmount']))
    # df = pd.DataFrame(X_train, columns = X_train.columns)
    # X_tsne_2 = TSNE(n_components=2).fit_transform(X)
    
    iForest =  IsolationForest(max_samples=400, 
                               n_estimators = 150,
                               contamination= 0.4, 
                               random_state=42)
    
    y_pred = iForest.fit_predict(X)
    
    df = df.copy()
    
    df['label'] = y_pred
    df['anomaly'] = pd.Series(y_pred)
    df['scores'] = iForest.decision_function(X)
    
    # print("\t--Counter: ", Counter(y_pred))
    
    # print("--- before :\n", y_pred, "\n")
    
    mapped_pred = mapPredLabelToLabel(y_pred, 1, -1)
    # print(mapped_pred)
    print("\t--Counter: ", Counter(mapped_pred))
    
    
    # print(classification_report(y_true=y, y_pred= mapped_pred, target_names=['normal class', 'anomaly class'], digits=4))
    
    res = retreiveScoreFromY(y=y, y_pred=y_pred)
    
    # disp = DecisionBoundaryDisplay.from_estimator(
    #     iForest,
    #     X,
    #     response_method="decision_function",
    #     alpha=0.5,
    # )
    # disp.ax_.scatter(X[:, 0], X[:, 1], c=y_pred, s=20, edgecolor="k")
    # disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
    # plt.axis("square")
    # plt.legend(labels=["outliers", "inliers"], title="true class")
    # plt.colorbar(disp.ax_.collections[1])
    # plt.show()
    
    return res
    
    # plotAnomalyGraph(df = df, field="uniq_receive_address_num")
    
    
    
    # -- Plot score graph distributiom ---
    # df['anomaly'] = df['label'].apply(lambda x: 'outlier' if x==-1  else 'inlier') 
    # fig = px.histogram(df,x='scores',color='anomaly') 
    # fig.show()
    
    
    # --- Plot Figures ---
    
    # disp = DecisionBoundaryDisplay.from_estimator(
    #     iForest,
    #     X_pca_2,
    #     response_method="decision_function",
    #     alpha=0.5,
    # )
    # disp.ax_.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y_pred, s=20, edgecolor="k")
    # disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
    # plt.axis("square")
    # plt.legend(labels=["outliers", "inliers"], title="true class")
    # plt.colorbar(disp.ax_.collections[1])
    # plt.show()



def strafiedCV(func, X_all, y_all, n = 10):
    sss = StratifiedShuffleSplit(n_splits=n, test_size=0.2, 
                                random_state=33) 
    
    sss.get_n_splits(X_all, y_all) 
    
    cv_score = {
        "Accuracy": [],
        "Precision": [],
        "Specificity": [],
        "Recall": [],
        "FPR": [],
        "FNR": [],
    }
    
    cnt = 1
    for train_index, test_index in tqdm(sss.split(X_all, y_all)): 
        # print(f"\n\t+++ Current Iteration -- {cnt} --\n")
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index] 
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index] 
        
        X_train_scaled = stdTrainData(X_train)
        X_test_scaled = stdTrainData(X_test)
        
        res = func(df=X_train, X=X_train_scaled, y=y_train)
        # print(res)
        
        for key in cv_score.keys():
            cv_score[key].append(res[key])
        
        cnt += 1
    
    print("### CV 10-Fold Stratified ###\n")
    for key in cv_score.keys():
        print(f"-- {key}: {round(mean(cv_score[key]), 3)}")
        
        

def retreiveScoreFromY(y, y_pred):
    acc_score = accuracy_score(y_true=y, y_pred=y_pred)
    prec_score = precision_score(y_true=y, y_pred=y_pred)
    spec_score = recall_score(y_true=y, y_pred=y_pred, pos_label=0)
    recall = recall_score(y_true=y, y_pred=y_pred)
    
    # 
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    
    fpr_score = (fp) / (fp + tn)
    fnr_score = (fn) / (fn + tp)
    # 

    res = {
        "Accuracy": acc_score,
        "Precision": prec_score,
        "Specificity": spec_score,
        "Recall": recall,
        "FPR": fpr_score,
        "FNR": fnr_score,
    }
    
    
    
    return res
    


def plotAnomalyGraph(df, field):
    # field = "uniq_receive_address_num"
    a = df.loc[df['anomaly'] == 1, field]
    b = df.loc[df['anomaly'] == -1, field]
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.hist([a, b], bins = 50, stacked = True, color = ['#BBBBBB', 'red'] )
    plt.yscale('log', base=10)
    
    plt.title(f"Anomaly values in {field}")
    plt.ylabel("Count")
    plt.xlabel(f"Number of {field}")
    plt.show()



# Convert custom normal/ anomaly label to 0: normal & 1: illicit
def mapPredLabelToLabel(pred_labels: [], normal_lbl: int, anomaly_lbl: int):
    
    try:
        for i in range(len(pred_labels)):
            if (pred_labels[i] == normal_lbl):
                pred_labels[i] = 0
            elif (pred_labels[i] == anomaly_lbl):
                pred_labels[i] = 1
            else:
                raise Exception("ERROR???")
    
    except Exception as e:
        print(e)
            
    
    
    return pred_labels
    
    
    
    
def trainUnsupervisedModel(X, X_unsampled, y, X_test, y_test):
    
    
    # Demension Reduction
    
    # 0.95 EVR for n = 37, 0.999 for n = 52;  1.0 for n = 59
    X_used = PCA(n_components=37).fit_transform(X)
    
    # X_used = PCA(n_components=37).fit_transform(X_unsampled)
    
    
    X_pca_3 = PCA(n_components=3).fit_transform(X)

    
    # X_test_pca = PCA(n_components=3).fit_transform(X_test)
    
    # T-SNE
    # X_tsne_3 = TSNE(n_components=3).fit_transform(X)
    
    
    # ---- Hyperparameter Setting -----
    
    # optHyperParm(modelName="KMeans", modelFunc=KMeans, X=X_used)
    # # Best k = 5
    # return
    
    # optHyperParm(modelName="Gaussian Mixture Model clustering", modelFunc=GaussianMixture, X=X_used)
    # # Best k = 3-4
    
    # optHyperParm(modelName="Hierarchical Cluster", modelFunc = AgglomerativeClustering, X = X_used)
    # # Best k = 4-9
    
    # optHyperParm(modelName="DBSCAN", modelFunc = DBSCAN, X = X_used)
    
    # Best: min_cluster = 345 to 435 (select 390), min_samples = 9
    # optHyperParm(modelName="HDBSCAN", modelFunc = HDBSCAN, X = X_used)
    

    
    # return
    
    
    # ---- TesTing AREA
    
    # kmeans = KMeans(n_clusters=7, init='random', n_init = "auto", random_state=5)
    # pred_y = Evaluate("Kmeans", kmeans, X_used)
    # plot_3D_graph(title = "K-means's 3D graph (PCA)", x = X_used, y_clusters = pred_y)
    # # print(pred_y)
    
    # kmeans = KMeans(n_clusters=7, init='random', n_init = "auto", random_state=5)
    # pred_y = Evaluate("Kmeans", kmeans, X_tsne)
    # plot_3D_graph(title = "K-means's 3D graph (t-SNE)", x = X_tsne, y_clusters = pred_y)
    
    # return
    
    # saveClustersInfo(y=y, pred_y=pred_y)
    # return
    
    
    # TODO: GMMs
    # Perform clsutering
    # gm = GaussianMixture(
    #                     n_components = 4,
    #                     n_init = 10)
                        
    # # Train the algorithm
    # pred_y = Evaluate(gm, X_used)
    
    # plotClustering(name="Gaussian Mixture Model", data=X_used, model=gm, pred_y=pred_y)
    # plot_3D_graph(title = "GMM's 3D graph", x = X_used, y_clusters = pred_y)
    
    
    # return
    
    # kmeans = KMeans(n_clusters=2, init='k-means++', random_state=5)
    # pred_y = Evaluate(kmeans, X_used)
    
    # optHyperParm(modelName="DBSCAN", modelFunc = DBSCAN, X = X_used)
    
    # Best: min_cluster = 345 to 435 (select 390), min_samples = 9
    # optHyperParm(modelName="HDBSCAN", modelFunc = HDBSCAN, X = X_used)

    # return

    # ---------------------
    
    
    # Initialize the clustering models
    # 7-8 n 
    kmeans = KMeans(n_clusters=7, init='random', n_init = "auto", random_state=5)
    
    # 7 n --> beomce
    hierarchical_cluster = AgglomerativeClustering(n_clusters=7, metric='euclidean', linkage='ward')
    
    # Best: 4 clusters
    dbscan = DBSCAN(eps=0.951, min_samples=9).fit(X_used)
    
    # Best: min_cluster = 380 to 580 (select 480), min_samples = 4
    hdbscan = HDBSCAN(min_cluster_size = 480, min_samples = 4).fit(X_used)
    
    
    gm = GaussianMixture(
                        n_components = 4,
                        n_init = 10)
    
    cluster_models = [
        kmeans,
        hierarchical_cluster,
        dbscan,
        hdbscan,
        gm,
    ]
    
    models_name = [
        "K-Means",
        "Hierarchical Cluster",
        "DBSCAN Cluster",
        "HDBSCAN Cluster",
        "Gaussian Mixture",
    ]
    
    
    headerArr = [
        "Model",
        "n_clusters",
        "outliners",
        "Silhouette",
        "Davies_Bouldin",
    ]
    
    
    
    # Separate TEST models
    curModelName = "Hierarchical Cluster"
    
    # 0.95 EVR
    pred_y, cnt = Evaluate(curModelName, hierarchical_cluster, X_used)
    print(cnt)
    # saveClustersInfo(modelName = curModelName, y = y, pred_y = pred_y)
    # return

    # 0.95 EVR
    pred_y, cnt = Evaluate("K-means", kmeans, X_used)
    print(cnt)
    
    
    # plot Hier Cluster graph
    # hier_deno = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric='euclidean', linkage='ward')
    # hier_deno = hier_deno.fit(X_used)
    # plot_dendrogram(hier_deno, truncate_mode="level", p=3)
    
    
    # PCA
    # pred_y_pca = Evaluate(curModelName, hierarchical_cluster, X_pca_3)
    # plot_3D_graph(title = f"{curModelName}'s clustering 3D graph (PCA)", x = X_pca_3, y_clusters = pred_y_pca) 
    
       
    # t-SNE
    # pred_y_tsne = Evaluate(curModelName, hierarchical_cluster, X_tsne_3)
    # plot_3D_graph(title = f"{curModelName}'s clustering 3D graph (t-SNE)", x = X_tsne_3, y_clusters = pred_y_tsne) 
    
    
    # return
    
    
    
    
    print(sepChar.join(headerArr))
    for model, name in zip(cluster_models, models_name):
        # print(f"### {name} ###")
        pred_y, cnt = Evaluate(name, model, X_used)
        # plotClustering(name=name, data=X_used, model=model, pred_y=pred_y)
        
    
    return
    
    # -------------------    



def PCA_evaluate(X_pca, model = KMeans):
    # /var/folders/rw/ccy6cl5s3fdc1xckcrfhrrqw0000gn/T/com.apple.useractivityd/shared-pasteboard/items/A33D960E-4924-470F-A3EF-A369FBB3F811/8697e05eee9ec49574b58922b6471d6367ece4e1.rtfd
    # n_components=None, 主成分个数默认等于特征数量
    pca = PCA(n_components=None)

    # 拟合数据
    pca.fit(X_pca)

    # 获取解释方差比率
    evr = pca.explained_variance_ratio_ * 100
    
    print(np.arange(1, len(evr) + 1), "\n", np.cumsum(evr))
    
    # 
    res = []
    
    for i in np.arange(1, len(evr) + 1):
        res.append({"Components": i, "EVR": np.cumsum(evr)[i - 1]})
    
    df = pd.DataFrame(res)
    df.to_csv("pca_test_unsampled.csv")


    # 查看累计解释方差比率与主成分个数的关系
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, len(evr) + 1), np.cumsum(evr), "-b")
    ax.set_title("PCA Cumulative Explained Variance Ratio")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained variance ratio(%)")
    fig.show()
    plt.show()

def plotClustering(name, data, model, pred_y):
    # Getting unique labels
    u_labels = np.unique(pred_y)
    
    # --- plotting the results:
    # -- 2D graphs
    for i in u_labels:
        plt.scatter(data[pred_y == i , 0] , data[pred_y == i , 1] , label = i, s= 30, alpha=0.8)
    
    # plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='^', color='red', linewidths=2)

    plt.title(f"Clustering of address from {name} Algorithms", fontsize = 15)
    plt.legend()
    plt.show()


def Evaluate(name, model, data):
    
    # model.fit(data)
    # predict the labels of clusters.
    pred_y = model.fit_predict(data)
    # labels = model.labels_
    
    # --- Evaluation metrics
    silh_score = silhouette_score(data, pred_y)
    # print(f"\t-- Silhouette score: {silh_score} --")
    
    # Compute Davies-Bouldin index
    davies_bouldin = davies_bouldin_score(data, pred_y)
    # print(f"\t-- Davies-Bouldin index: {davies_bouldin} --\n")
    
    #Count the number of clusters under each parameter combination (- 1 indicates abnormal point)
    n_clusters = len([i for i in set(pred_y) if i != -1])
    
    #Number of outliers
    outliners = np.sum(np.where(pred_y == -1, 1, 0))
    
    
    # print Label Distribution
    cnt = Counter(pred_y)
    # print("\t--Counter: ", Counter(pred_y))
    
    resArr = [
        name,
        n_clusters,
        outliners,
        silh_score,
        davies_bouldin
    ]
    
    resArr = [str(x) for x in resArr]
    
    print(sepChar.join(resArr))
    
    return pred_y, cnt


def optHyperParm(modelName, modelFunc, X):
    elbow_methods = [] 
    
    # 使用輪廓分析法找到最佳的集群數
    silhouette_scores = []
    
    maxCluster = 10
    
    if (modelFunc == DBSCAN):
        #Build an empty list to save the results under different parameter combinations
        res = []
        for eps in tqdm(np.arange (0.001 , 1 , 0.05)) :
            #Iterating different values of min_samples 
            for min_samples in range (2, 10):
                
                try:
                    print(f"Testing [eps: {eps}, min_samples: {min_samples}]...")
                    dbscan = DBSCAN(eps = eps , min_samples = min_samples)

                    y_pred = dbscan.fit_predict(X)
                    sil_score = (silhouette_score(X, y_pred))
                    davies_bouldin = davies_bouldin_score(X, y_pred)
                    
                    
                    
                    #Count the number of clusters under each parameter combination (- 1 indicates abnormal point)
                    n_clusters = len([i for i in set(dbscan. labels_) if i != -1])
                    
                    #Number of outliers
                    outliners = np.sum(np.where(dbscan. labels_ == -1, 1, 0))
                    
                    #Count the number of samples in each cluster
                    stats = str(pd.Series([i for i in dbscan. labels_ if i != -1]).value_counts().values)
                    
                    # print(f"'sil_score': {sil_score}, 'davies_score': {davies_bouldin}, 'eps':{eps} , 'min_samples': {min_samples} , '_clusters': {n_clusters} , 'outliners': {outliners}, 'stats': {stats}")
                    res.append ({'sil_score': sil_score, 'davies_score': davies_bouldin, 'eps':eps , 'min_samples':min_samples , '_clusters':n_clusters , 'outliners':outliners, 'stats': stats})
        
                except Exception as e:
                    print("ERROR!!!: ", e)
                    res.append ({'sil_score': "NaN", 'davies_score': "NaN", 'eps':eps , 'min_samples': min_samples , '_clusters': "NaN" , 'outliners': "NaN", 'stats': "NaN"})


        
        df = pd.DataFrame(res)
        df.to_csv("DBSCAN_hyperparam_new.csv")
        
        return
    
    
    if (modelFunc == HDBSCAN):
        res = []
        
        for min_size in tqdm(np.arange (300 , 600 , 20)):
            
            for min_samples in range (4, 10):
                
                try:
                    print(f"Testing [min_cluster_size: {min_size}, min_samples: {min_samples}]...")
                    hdbscan = HDBSCAN(min_cluster_size = min_size, min_samples = min_samples)

                    y_pred = hdbscan.fit_predict(X)
                    sil_score = (silhouette_score(X, y_pred))
                    davies_bouldin = davies_bouldin_score(X, y_pred)
                    
                    #Count the number of clusters under each parameter combination (- 1 indicates abnormal point)
                    n_clusters = len([i for i in set(hdbscan. labels_) if i != -1])
                    
                    #Number of outliers
                    outliners = np.sum(np.where(hdbscan.labels_ == -1, 1, 0))
                    
                    #Count the number of samples in each cluster
                    stats = str(pd.Series([i for i in hdbscan.labels_ if i != -1]).value_counts().values)
                    
                    # print(f"'sil_score': {sil_score}, 'davies_score': {davies_bouldin}, 'eps':{eps} , 'min_samples': {min_samples} , '_clusters': {n_clusters} , 'outliners': {outliners}, 'stats': {stats}")
                    res.append ({'sil_score': sil_score, 'davies_score': davies_bouldin, 'min_cluster_size': min_size, 'min_samples': min_samples , '_clusters':n_clusters , 'outliners':outliners, 'stats': stats})

                except Exception as e:
                    print("ERROR!!!: ", e)
                    res.append ({'sil_score': "NaN", 'davies_score': "NaN", 'min_cluster_size': min_size, 'min_samples': min_samples , '_clusters': "NaN" , 'outliners': "NaN", 'stats': "NaN"})
                    try:
                        df = pd.DataFrame(res)
                        df.to_csv(f"tmp_{min_samples}-{min_size}HDBSCAN_hyperparam_pca_52.csv")
                    
                    except Exception as ee:
                        print("Antoher ERR! ", ee)
                    
                    finally:
                        continue
                
        df = pd.DataFrame(res)
        path_file = "HDBSCAN_hyperparam_new_pca_52.csv"
        df.to_csv(path_file)
        print("Saving to " + path_file + "...")
        return
    
    
    for n in range(2, maxCluster + 1):
        
        print(f"Testing cluster# = {n}")
        
        if (modelFunc == KMeans):   
            clustering = KMeans(n_clusters = n, init='k-means++')
            clustering.fit(X)
            elbow_methods.append(clustering.inertia_)

            kmeans = KMeans(n_clusters = n, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        
        elif (modelFunc == AgglomerativeClustering):
            ac = AgglomerativeClustering(n_clusters = n, affinity='euclidean', linkage='ward')
            y_pred = ac.fit_predict(X)
            # elbow_methods.append(ac.inertia_)
            silhouette_scores.append(silhouette_score(X, y_pred))
        
        elif (modelFunc == GaussianMixture):
            gmms = GaussianMixture(
                        n_components = n,
                        n_init = 10)
            gmms_labels = gmms.fit_predict(X)
            # elbow_methods.append(gmms.inertia_)
            silhouette_scores.append(silhouette_score(X, gmms_labels))
            
            print("sil score: ", silhouette_scores)


    # for finding optimal no of clusters we use elbow technique 
    # Elbow technique is plot between no of clusters and objective_function 
    # we take k at a point where the objective function value have elbow shape 
    
    y_to_plot = [elbow_methods, 
                 silhouette_scores]
    
    titles = [f'The Elbow Method for {modelName}', 
              f'Silhouette Analysis of {modelName}']
    
    y_lbls = ['Sum of the squared errors (SSE)', 
              'Silhouette Score']
    
    for i in range(1, 2):
        plt.plot(range(2, maxCluster + 1), y_to_plot[i])
        plt.title(titles[i], fontsize=18)
        
        plt.xticks(range(2, maxCluster + 1, 2))
        plt.xlabel('Number of Clusters', fontsize=16)
        plt.ylabel(y_lbls[i], fontsize=16)
        plt.show()
    
    
    # 繪製輪廓分析法圖



def saveClustersInfo(modelName, y, pred_y):
    y = y.to_numpy()
    print(y)
    
    res = []
    
    n_clusters = len([i for i in set(pred_y) if i != -1])
    
    for i in range(len(y)):
        res.append({"flag": y[i], "cluster_gp": pred_y[i]})
    
    df = pd.DataFrame(res)
    df.to_csv(f"{modelName}_cluster_info.csv")
    
    print(f"\n--Saved to {modelName}_cluster_info.csv...\n")
    
    
    

if __name__ == "__main__":
    main()