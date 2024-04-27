
import numpy as np
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve

# 
from scipy.cluster.hierarchy import dendrogram



SMALL_SIZE = 8
STANDARD_SIZE = 16
TITLE_SIZE = 18

plt.rc('font', size=STANDARD_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title



def plotDataDistribution(y, title):
    # Count the number of samples in each class
    class_counts = np.bincount(y)
    data_cnt = y.shape[0]
    
    palette_color = sns.color_palette('bright') 
    
    # Create a bar plot to visualize the distribution of target classes
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    
    label = [0, 1]
    txt_label = ["Normal", "Illicit"]
    
    
    data = [ y.loc[lambda x : x == 0].shape[0], y.loc[lambda x : x == 1].shape[0] ]
    
    wedges, texts, autotexts = ax.pie(x = data, labels = txt_label, autopct='%.1f%%', textprops={'fontsize': 14})
    
    
    for autotext in autotexts:
      autotext.set_color('white')
    
    # Adding legend
    ax.legend(wedges, txt_label,
          title ="Types",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1), fontsize=16)
    
    
    # ax = sns.barplot(x=np.unique(y), y=class_counts)
    # ax.bar_label(ax.containers[0], fontsize=10)
    # for container in ax.containers:
    #     ax.bar_label(container /100, fmt='%.1f%%', fontsize=10)
    
    
    # plt.xlabel('Type', fontsize=16)
    # plt.xlabel('''Type
    #            Note: type 0 refer to normal accounts and type 1 refer to illicit accounts''')
    # plt.ylabel("Percentage", fontsize=16)
    
    
    plt.title(title, fontsize=18)
    plt.show()
    
    
    
def plotROC_curve(title, estimator, X_test, y_test):
  RocCurveDisplay.from_estimator(estimator, X_test, y_test)
  
  plt.xlabel(xlabel="False Positive Rate (Positive label: 1)", fontsize=16)
  plt.ylabel(ylabel="True Positive Rate (Positive label: 1)", fontsize=16)
  plt.title(title, fontsize=18)
  
  plt.show()


def plotROC_curve_pred(title, method, y_test, y_pred, isShow = True):
  # RocCurveDisplay.from_predictions(y_test, y_pred)
  
  fpr, tpr, threshold = roc_curve(y_test, y_pred)
  
  auc = round(roc_auc_score(y_test, y_pred), 4)
  plt.plot(fpr, tpr, label=f"{method}, AUC = "+str(auc))
  
  plt.xlabel(xlabel="False Positive Rate (1 - Speciticity)", fontsize=16)
  plt.ylabel(ylabel="True Positive Rate (Sensitivity)", fontsize=16)
  
  if (isShow):
    #add legend
    plt.title(title, fontsize=18)
    plt.legend()
    plt.show()
  
  
  
def plot_3D_graph(title, x, y_clusters):
  
  # Getting unique labels
  u_labels = np.unique(y_clusters)
  
  
  fig = plt.figure(figsize = (8,8))
  ax = fig.add_subplot(111, projection='3d')
  
  for i in u_labels:
    if (i == -1):
      continue
    ax.scatter(x[y_clusters == i,0],x[y_clusters == i,1],x[y_clusters == i,2], s = 40 , label = i)

    
    
  ax.set_xlabel('First Components')
  ax.set_ylabel('Second Components')
  ax.set_zlabel('Third Components')
  ax.legend()
  
  plt.title(title)
  plt.show()
  
  
  
from matplotlib.patches import Ellipse

def plot_gmm(gmm, X, label=True, ax=None):
    def draw_ellipse(position, covariance, ax=None, **kwargs):
        ax = ax or plt.gca()
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
            for nsig in range(1, 4):
                ax.add_patch(Ellipse(
                    position, nsig * width, nsig * height,
                    angle, **kwargs))
    
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=7, cmap='viridis', zorder=2) 
    else:
        ax.scatter(X[:, 0], X[:, 1], s=7, zorder=2)
    
    ax.axis('equal')
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance")
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()