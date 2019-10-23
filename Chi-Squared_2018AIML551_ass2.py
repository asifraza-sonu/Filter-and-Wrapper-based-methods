#Chi square filter method
import pandas as pd
from sklearn.feature_selection import chi2
X = pd.read_csv("discretized_glass_features.csv")
X = X.drop([X.columns[0]], axis=1)
X = X.astype(int)
y = pd.read_csv("glass_target.csv")
chi2_scores, p_values= chi2(X,y)
data = []
for i in range(9):
    data.append([X.columns[i],chi2_scores[i],p_values[i]])
DF1= pd.DataFrame(data,columns = ['Feature','Chi square value','P value'])
DF2 = DF1.sort_values(by='Chi square value', ascending=False)
print ('a:The most significant feature to least significant feature is of below order')
print(DF2)


print('The ranking of score of feature to target label , can be given from 1 to 9 (highest to lowest)')
j = 0
for j in range(0,9):
        print ('Feature:',DF2['Feature'].values[j],'     :','Corresponding''Rank of significance is :',j+1)
#If ever we have been asked to select best 5 features out of given 9 features
DF3 = DF2.head()
print('The best five features that can be taken are ',DF3['Feature'].values)
print('b.Value of Chi-squared ')
i = 0
for i in range(0,9):
        print ('Feature:',DF2['Feature'].values[i],'Corresponding','     :','CHi-squared value is :',DF2['Chi square value'].values[i])
        
    
        
    