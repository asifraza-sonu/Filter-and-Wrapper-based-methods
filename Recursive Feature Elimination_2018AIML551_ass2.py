from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

print("\n\nWrapper-based Method (using K-Nearest Neighbor classifier as the underlying classification algorithm)\n")
X = pd.read_csv("glass_features.csv")
y = pd.read_csv("glass_target.csv")

#Lets try to put the number of nearest neighbors from in the range of given number of features,because depending upon this the KNN will compare the values
for n in range (1,10):
    print('When The Number of nearest neighbors selected are' ,n)
    knn = KNeighborsClassifier(n_neighbors=n)
    # the param forward when set to False will do sequential backward selectioni.e recursive feature elimination
    #Also since we have provided the string "best" in K_features , as per the docstring it will give us the best subset which is having best cross validation score
    sbs = SFS(knn,
           k_features='best',
           forward=False,
           scoring='accuracy')
    sbs = sbs.fit(X, y)
    print("Best  features Subset by SFS for this KNN algorithm when selected number of neighbors are : ",n,'are :',sbs.k_feature_idx_)
    Data2 = []
    for ig in sbs.k_feature_idx_:
        k = int(ig)
        print(k)
        Data2.append(X.columns[k])
    new_X = X[Data2]
    knn.fit(new_X,y)
    y_predict = knn.predict(new_X)
    #print ('The Corresponding R2 value of Nearest neighbor selected :',n,'is :',knn.score(new_X,y_predict))
    print ('The Corresponding RMSE valueof Nearest neighbor selected :',n,'is :',math.sqrt(mean_squared_error(y,y_predict)))
    
    

    #best nearest neighbors numbers are : 4,2


