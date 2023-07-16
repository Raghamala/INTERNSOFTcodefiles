# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 14:19:57 2023

@author: Ragha
"""


#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt



#READING THE DATA FROM YOUR FILES
data = pd.read_csv("advertising.csv")
data.head()



#to visualize data
fig , axs = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])




#creating X&Y for linear regression
feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales



#importing linear regression algo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



result = 6.974821488229891 + 0.05546477*50
print(result)



#create a dataframe with min and max value of the table
X_new = pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()



preds = lr.predict(X_new)
preds 




data.plot(kind = 'scatter', x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth=3)




import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lm.conf_int()



#finding the probability values
lm.pvalues



#finding the R-Squared values
lm.rsquared



#multi linear regression
feature_cols = ['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales 



lr = LinearRegression()
lr.fit(X,y)


print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper', data=data).fit()
lm.conf_int()
lm.summary()