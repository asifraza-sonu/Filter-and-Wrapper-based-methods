#Information gain filter method
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
X = pd.read_csv("glass_features.csv")

y = pd.read_csv("glass_target.csv")
y= y.values.ravel()
mutual_info_gain_values = mutual_info_classif(X,y, discrete_features=False,random_state=42)
data = []
for i in range(9):
    
    data.append([X.columns[i],mutual_info_gain_values[i]])

df = pd.DataFrame(data, columns = ['feature','info_gain'])
DF2 = df.sort_values("info_gain", axis = 0, ascending = False)

print ('a:The most significant feature to least significant feature is of below order')
print(DF2)


print('The ranking of score of feature to target label , can be given from 1 to 9 (highest to lowest)')
j = 0
for j in range(0,9):
        print ('Feature:',DF2['feature'].values[j],'     :','Corresponding''Rank of significance is :',j+1)
#If ever we have been asked to select best 5 features out of given 9 features
DF3 = DF2.head()
print('The best five features that can be taken are ',DF3['feature'].values)
print('b.Value of Information gain')

i = 0
for i in range(0,9):
        print ('Feature:',DF2['feature'].values[i],'Corresponding','     :','Value of Information gain :',DF2['info_gain'].values[i])
        
    
    

