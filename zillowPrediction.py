#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:12:54 2018
pip
@author: matttroglia
"""

# import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import (LassoCV , LinearRegression, Ridge,Lasso,RandomizedLasso)
from sklearn.preprocessing import (StandardScaler,MinMaxScaler)
from sklearn.feature_selection import RFE,f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

#from ggplot import *

# %%
zillowData = 'ZillowDataAll/properties_2016.csv'
zillowTrain2016 = 'ZillowDataAll/train_2016_v2.csv'
zillowDF = pd.read_csv(zillowData)
train = pd.read_csv(zillowTrain2016, parse_dates=['transactiondate'])
zillowDF.hashottuborspa = zillowDF.hashottuborspa.astype(str)
zillowDF.fireplaceflag = zillowDF.fireplaceflag.astype(str)

zillowTest = 'ZillowDataAll/train_2017.csv'
zillowTestData = 'ZillowDataAll/properties_2017.csv'
zillowDF_Test = pd.read_csv(zillowTestData)
test = pd.read_csv(zillowTest, parse_dates=['transactiondate'])
zillowDF_Test.hashottuborspa = zillowDF.hashottuborspa.astype(str)
zillowDF_Test.fireplaceflag = zillowDF.fireplaceflag.astype(str)


for col in zillowDF.columns:
    if zillowDF[col].dtype == 'object':
        print(col)
        lbl = LabelEncoder()
        lbl.fit(list(zillowDF[col].values))
        zillowDF[col] = lbl.transform(list(zillowDF[col].values))

for col in zillowDF_Test.columns:
    if zillowDF_Test[col].dtype == 'object':
        print(col)
        lbl = LabelEncoder()
        lbl.fit(list(zillowDF_Test[col].values))
        zillowDF_Test[col] = lbl.transform(list(zillowDF_Test[col].values))

# %%

# DETERMINE IF WE HAVE LOG ERRORS THAT ARE OUTLIERS
plt.figure(figsize=(8, 6))
plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))
plt.show()

upperLim = np.percentile(train.logerror.values, 99)
lowerLim = np.percentile(train.logerror.values, 1)

train['logerror'].loc[train['logerror'] > upperLim] = upperLim
train['logerror'].loc[train['logerror'] < lowerLim] = lowerLim
#%%
plt.figure(figsize=(8, 6))
plt.scatter(range(train.shape[0]), np.sort(train.logerror.values))
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(train.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()




#%%
zillowDF.shape
missing_df = zillowDF.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

missing_df_test = zillowDF_Test.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test = missing_df.ix[missing_df['missing_count']>0]
missing_df_test = missing_df.sort_values(by='missing_count')


#%%Dislay the features that are missing data
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()

#%%
'''
plt.figure(figsize=(12,12))
sns.jointplot(x=zillowDF.latitude.values, y=zillowDF.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
'''
#%%

train = pd.merge(train, zillowDF, on='parcelid', how='left')
train.head()

test = pd.merge(test, zillowDF_Test, on='parcelid', how='left')
test.head()
#%%

missing_df = train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]


missing_df_test = train.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio'] = missing_df_test['missing_count'] / train.shape[0]
missing_df_test.ix[missing_df_test['missing_ratio']>0.999]




#%%

#mean_values = train.mean(axis=0)

#%%
trainTypes = train.dtypes.reset_index()
trainTypes.columns = ["item","Type"]

trainTypes.groupby("Type").aggregate('count').reset_index()

#%%
#ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) + \
 #   geom_point() + \
  #  scale_color_gradient(low = 'red', high = 'blue')

#%%
#remove some of the data that is not needed for learning
train_y= train['logerror'].values
catagory_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_x = train.drop(['parcelid', 'logerror', 'transactiondate']+catagory_cols, axis=1)

test_y = test['logerror'].values
test_x = test.drop(['parcelid', 'logerror', 'transactiondate']+catagory_cols, axis=1)
#%%
mean_values = train_x.mean(axis=0)
#need to do something with NAN Values
#train _x = train_x.fillna(0)
train_x= train_x.fillna(mean_values)
feature_names = train_x.columns.values

mean_values_test = test_x.mean(axis=0)
test_x = test_x.fillna(mean_values_test)

#%% Get feature rankings

ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))




# Randomized Lasso works by resampling the train data and computing a Lasso on each resampling.
# The features selected more often are good features. It is also known as stability selection
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(train_x, train_y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), feature_names)
print('finished')

#####

# Ordinary Least Squares Linear Regression model J = 1/2 sum|O-y|^2
linrreg = LinearRegression(normalize=True)
linrreg.fit(train_x,train_y)

#Feature ranking with recursive feature elimination. stop the search when only the last feature is left
'''
the goal of recursive feature elimination (RFE) is to select features by recursively
considering smaller and smaller sets of features. First, the estimator is trained on the
initial set of features and the importance of each feature is obtained either through a
coef_ attribute or through a feature_importances_ attribute. Then, the least important
features are pruned from current set of features. That procedure is recursively repeated
 on the pruned set until the desired number of features to select is eventually reached.
'''
rfe = RFE(linrreg, n_features_to_select=1, verbose =3 )
rfe.fit(train_x,train_y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), feature_names, order=-1)



#Get the weight vectors for three common types of regression problems
# Using Linear Regression
linrreg = LinearRegression(normalize=True)
linrreg.fit(train_x,train_y)
ranks["LinReg"] = ranking(np.abs(linrreg.coef_), feature_names)

# Using Ridge
ridge = Ridge(alpha = 7)
ridge.fit(train_x,train_y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), feature_names)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(train_x,train_y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), feature_names)


#Get feature importance in a random forese regressor.

rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=2)
rf.fit(train_x,train_y)
ranks["RF"] = ranking(rf.feature_importances_, feature_names)

####

# Create empty dictionary to store the mean value calculated from all the scores
# keys dict_keys(['rlasso/Stability', 'RFE', 'LinReg', 'Ridge', 'Lasso', 'RF'])
r = {}
#find mean of each features by regression/learning type and round all elements to 2 decimal places
for name in feature_names:
    r[name] = round(np.mean([ranks[method][name]for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

#print matrix of the each learning type
print("\t%s" % "\t".join(methods))
for name in feature_names:
    print("%s\t%s" % (name, "\t".join(map(str,[ranks[method][name] for method in methods]))))


# Dataframe of the mean scores of learning type with features
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe by mean ranking
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

# Let's plot the ranking of the features
sns.catplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", height=10,
               aspect=1, palette='coolwarm')
plt.show()
#%%

#get top n perfroming features and train on it, Meanplot needs to be sorted first
top_features= meanplot['Feature'][0:14]
top_train_data = train[top_features]
mean_values = top_train_data.mean(axis=0)
#need to do something with NAN Values
#train _x = train_x.fillna(0)
top_train_data= top_train_data.fillna(mean_values)
#top_train_data.values()

Model_Lasso = Lasso()
Model_Lasso.fit(top_train_data,train_y)
print('Lasso Predicting...')
#y_pred_lasso= Model_Lasso.predict(top_train_data)
print('Getting Training Error...')
print('Training Error: ',Model_Lasso.score(top_train_data,train_y))
print('Test Error: ', Model_Lasso.score(test_x[top_features],test_y))
#%%
Model_Ridge = Ridge()
Model_Ridge.fit(top_train_data,train_y)
print('Ridge Predicting...')
#y_pred_lasso= Model_Lasso.predict(top_train_data)
print('Getting Training Error...')
print('Training Error: ',Model_Ridge.score(top_train_data,train_y))
print('Test Error: ', Model_Ridge.score(test_x[top_features],test_y))
#%%
Model_svm = svm.SVR(kernel='rbf', gamma='auto',C=1)
Model_svm.fit(top_train_data,train_y)
print('Ridge Predicting...')
#y_pred_lasso= Model_Lasso.predict(top_train_data)
print('Getting Training Error...')
y_pred = Model_svm.predict(test_x[top_features])
#print('Training Error: ',Model_svm.score(top_train_data,train_y))
#print('Test Error: ', Model_svm.score(test_x[top_features],test_y))

sklearn.metrics.mean_squared_error(test_y, y_pred)
