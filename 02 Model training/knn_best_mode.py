
class KNN_param:
  
  def __init__(self, accuracy: float, weight: str, k: int, p: int):
    self.accuracy = accuracy
    self.weight = weight
    self.k = k
    self.p = p
    
  def __str__(self):
    return f"\tAccuracy: {self.accuracy}\n\tWeight: '{self.weight}'\n\tk: {self.k}\n\tp: {self.p}\n)"

def main():



  ## 設定欲找尋的k值範圍
  # best_param = KNN_param(0, "", -1, -1)
  # k_value_range = range(3,50)
  # weights_li = ['uniform', 'distance']
  
  # ## 裝測試結果的平均分數
  # k_value_scores = []
  
  # for k in tqdm(k_value_range):
    
      # for w in weights_li:
      # if (w == "uniform"):
      #     
      
      
  # for p in tqdm(range(1,7)):
    
  #         knn_model = KNeighborsClassifier(n_neighbors = k, weights = 'uniform', algorithm="auto", leaf_size=30, p = 2, metric="minkowski", metric_params=None)
  #         accuracy = cross_val_score(knn_model, X, y, cv=10, scoring="accuracy")
  #         # print("testing K值: ", k, "...")
  #         # print("Accuracy: ", accuracy.mean())
  #         k_value_scores.append(accuracy.mean())
  #         avg_accuracy = float(accuracy.mean())
          
  #         if (avg_accuracy > best_param.accuracy):
  #             best_param = KNN_param(avg_accuracy, 'uniform', k, -1)
          
          
  #         # if (w == "uniform"):
  #         #   break
          
      
  # print(f"Best KNN param:\n{best_param}")
  
  
  
  ## 找一個最佳的k值，由於k的初始值我們設在3，所以要加三
  # print("最佳K值: " ,k_value_scores.index(max(k_value_scores))+3)
  
  # plt.plot(k_value_range, k_value_scores, marker = "o")
  # plt.title("Best k-values in KNN")
  # plt.xlabel('K values')
  # plt.ylabel('Accuracy')
  # plt.show()
  
  
    # test draw figure
  # model = knnc
  # model.fit(X=X_train, y= y_train)
  # y_pred = model.predict(X_test)
  
  # plt.figure(figsize=(8, 8))
  # plt.rcParams['font.size'] = 14
  # plt.title(f'KNN (accuracy={accuracy_score(y_true=y_test, y_pred = y_pred):.1f}%)')
  #       # 畫出訓練集資料
  #       # 預測標籤 (大圓)
  # plt.scatter(*X_test.T, c = y_pred, cmap='tab10', s=100, alpha=0.8)
  #       # 實際標籤 (小圓)
  # plt.scatter(*X_test.T, c=y_test, cmap='Set3', s=35, alpha=0.8)
  # plt.grid(True)
  # plt.xlim([np.amin(X_test.T[0]), np.amax(X_test.T[0])])
  # plt.ylim([np.amin(X_test.T[1]), np.amax(X_test.T[1])])
  # plt.tight_layout()
  # plt.show()
  
  # return
  
  # return