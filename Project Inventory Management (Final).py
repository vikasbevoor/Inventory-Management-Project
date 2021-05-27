#!/usr/bin/env python
# coding: utf-8

# # Inventory Management Project

# In[2]:


# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Importing the dataset
data = pd.read_csv(r"C:\Users\Admin\Downloads\prorevenue.csv",low_memory=False )
data.head()


# In[4]:


data.shape


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.columns


# In[8]:


# Number of unique values
for feature in data.columns:
    count = data[feature].nunique()
    print(feature, ": ", count)


# ## Exploratory Data Analysis

# In[9]:


data.isna().sum()


# #### There are no NA values in the dataset

# In[10]:


# Encoding the "store status"


# In[11]:


data["store status"] = pd.get_dummies(data["store status"], drop_first=True)


# In[12]:


data["store status"].value_counts()


# In[13]:


# Converting the multiple categories of generic holidays into '0' and '1'


# In[14]:


data["Generic Holiday"].unique()


# In[15]:


data.loc[data["Generic Holiday"] == 'a',"Generic Holiday"] = 1
data.loc[data["Generic Holiday"] == 'b',"Generic Holiday"] = 1
data.loc[data["Generic Holiday"] == 'c',"Generic Holiday"] = 1
data.loc[data["Generic Holiday"] == '0',"Generic Holiday"] = 0


# In[16]:


data["Generic Holiday"].unique()


# In[17]:


data["Generic Holiday"].value_counts()


# In[18]:


data["Generic Holiday"]= data["Generic Holiday"].astype(int)


# In[19]:


data.describe()


# ### Correlation matrix

# In[20]:


correlation = data.corr()
correlation


# In[21]:


plt.figure(figsize = (15,6))
sns.heatmap(correlation, cbar = True, cmap = "viridis")


# #### There is a high correlation between the, 
# #### 1. "Revenue" and "No of purchases" -> 89.6 % correlation 
# #### 2. "Revenue" and "Store status"  -> 67.8 % of correlation

# ### Univariate analysis

# In[22]:


numerical_features = [feature for feature in data.columns if data[feature].dtypes != "O" and len(data[feature].unique())>25]
print("Number of numerical features:", len(numerical_features))


# In[23]:


# Using histograms to analyze the distribution of numerical features
for feature in numerical_features:
    dataset = data.copy()
    dataset[feature].hist(bins=20)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title("Histogram of "+ feature)
    plt.show()


# #### Both "Revenue" and "Number of purchases" are right skewwed

# In[24]:


# Using the boxplots to identify the outliers in the dataset
for feature in numerical_features:
    data = dataset.copy()
    data.boxplot(column=feature)
    plt.title("Boxplot of "+ feature)
    plt.show()


# #### There are lot of outliers in the "Revenue" and "Number of purchases"

# In[25]:


# Discrete features
discrete_features = [feature for feature in data.columns if feature not in numerical_features]


# In[26]:


# Countplots for discrete features
for feature in discrete_features:
    sns.countplot(feature, data= data, palette='hls')
    plt.show()


# ### Bivariate analysis

# In[27]:


# Scatterplot of numerical features
plt.plot(data["Revenue"], data["No of purchases"], "bo")
plt.xlabel("Revenue")
plt.ylabel("No of purchases")
plt.title("Scatter plot of "+ "Revenue and No of purchases")
plt.show()


# In[28]:


plt.plot(data["Product type"], data["Revenue"], "bo")
plt.xlabel("Product type")
plt.ylabel("Revenue")
plt.title("Scatter plot of "+ "Product type and Revenue")
plt.show()


# #### Some products have high "Revenue" and some products have lower "Revenue"

# In[29]:


# Discrete feature bivariate analysis with "Revenue"


# In[30]:


for feature in discrete_features:
    dataset = data.copy()
    data.groupby(feature)["Revenue"].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("Revenue")
    plt.title(feature+" v/s Revenue")
    plt.show()


# ### Observations
# 1. When the promotion is applied the revenue is higher.
# 2. "Education Holiday" is not effecting the store status.
# 3. On 1st day of week the revenue is highest and it reduces gardually till weekend (6h day).
# 4. Store remains closed on 7th day or Sunday.

# In[31]:


# Cross tabulation plot


# In[32]:


pd.crosstab(data["DayOfWeek"], data["store status"]).plot(kind="bar")


# In[33]:


pd.crosstab(data["Generic Holiday"], data["store status"]).plot(kind="bar")


# In[34]:


pd.crosstab(data["DayOfWeek"], data["Promotion applied"]).plot(kind="bar")


# #### Promotion is applied only on the weekdays i.e from 1st day of week to 5th day of week

# ## Importing Product details dataset

# In[35]:


data2 = pd.read_csv(r"C:\Users\Admin\Downloads\productdetails.csv")
data2.head()


# In[36]:


data2.shape


# In[37]:


data2.describe()


# In[38]:


data2.info()


# In[39]:


data.isna().sum()


# ## Combining two datasets

# In[40]:


data_full = data2.merge(data, left_on = "product type", right_on = "Product type", how = 'right')


# In[41]:


data_full.head()


# In[42]:


data_full.tail()


# In[43]:


data_full.drop(["Unnamed: 0", "Product type"], axis = 1, inplace = True)


# In[44]:


data_full.head()


# In[45]:


data_full.shape


# In[46]:


data_full.columns


# ### Aggregating data based on "Day of Week"

# In[82]:


data_full.groupby(by=["DayOfWeek"])[["Revenue","No of purchases",'cost per unit', 'Time for delivery']].mean()


# ### Adding column "Number of inventory" by dividing "Revenue" and "Cost per unit" with 10% buffer

# In[89]:


data_full["No_of_inventory"] = (data_full["Revenue"] / data_full["cost per unit"])*1.10


# In[92]:


data_full["No_of_inventory"] = data_full["No_of_inventory"].astype(int)


# In[93]:


data_full.head()


# In[95]:


data_full.groupby(by=["DayOfWeek"])[["Revenue","No_of_inventory",'cost per unit', 'Time for delivery']].mean()


# In[96]:


data_full.info()


# ### Checking the product types with highest and lowest "Revenue"

# In[103]:


# Top 10 product types with highest revenue


# In[101]:


data_full.groupby(["product type"])[['cost per unit', 'Time for delivery', 'Revenue','No_of_inventory']].mean().sort_values("Revenue", ascending = False).head(10)


# In[105]:


# Top 10 product types with lowest revenue


# In[102]:


data_full.groupby(["product type"])[['cost per unit', 'Time for delivery', 'Revenue','No_of_inventory']].mean().sort_values("Revenue", ascending = True).head(10)


# In[97]:


data_full.columns


# ### Seperating input and output variables

# In[107]:


X = data_full.drop(columns= ['No of purchases','Education Holiday','No_of_inventory'])
y = data_full['No_of_inventory']


# In[108]:


X.head()


# In[109]:


y.head()


# ### Scaling the input data

# In[111]:


from sklearn.preprocessing import MinMaxScaler


# In[112]:


scaler=MinMaxScaler()


# In[113]:


scaler.fit(X)


# In[114]:


scaled_data = pd.DataFrame(scaler.transform(X),columns=X.columns)


# In[115]:


scaled_data.head()


# ### Splitting the data into train and test data

# In[116]:


from sklearn.model_selection import train_test_split


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 96)


# In[122]:


print(X_train.shape,"\n",
X_test.shape)


# ### Using OLS model to check p-values

# In[123]:


import statsmodels.api as sm
from statsmodels.api import OLS


# In[124]:


est = sm.OLS(y, X).fit()


# In[125]:


est.summary()


# #### All the p-values are less than 0.05, hence they are significant

# ### Using Random Forest regressor to find important features

# In[126]:


from sklearn.ensemble import RandomForestRegressor 


# In[127]:


regressor = RandomForestRegressor(n_estimators = 20, random_state = 0) 


# In[128]:


regressor.fit(X, y) 


# In[129]:


# Get numerical feature importances
importances = list(regressor.feature_importances_)


# In[130]:


feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# In[131]:


# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# ## Linear regression

# In[132]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[133]:


reg = LinearRegression()


# In[135]:


model = reg.fit(X_train, y_train)


# In[136]:


print('Intercept: ', model.intercept_)
print('Coefficients: \n', model.coef_)


# In[138]:


# predicting trained values
y_pred1 = model.predict(X_train)


# In[139]:


# predicting test values
y_pred2 = model.predict(X_test)


# In[140]:


# Observed values v/s fitted values
plt.scatter(y_train, y_pred1, c="b");plt.xlabel("Observed values");plt.ylabel("Fitted values")


# In[ ]:


# RMSE value


# In[141]:


print("RMSE for training data: ",np.sqrt(metrics.mean_squared_error(y_train,y_pred1)))


# In[142]:


print("RMSE for testing data: ",np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))


# ## Random Forest Regressor

# In[143]:


from sklearn.ensemble import RandomForestRegressor 


# In[144]:


regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)


# In[145]:


regressor.fit(X_train, y_train) 


# In[146]:


# Predicting the values using the model


# In[147]:


Y_pred1 = regressor.predict(X_train) 


# In[148]:


Y_pred2 = regressor.predict(X_test) 


# In[149]:


# plot predicted data 
plt.plot(y_test,Y_pred2,'bo')  
plt.title('Random Forest Regression') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[150]:


# RMSE values


# In[151]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_train, Y_pred1)))


# In[152]:


print('RMSE of testing data:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred2)))


# ## XG Boost Regressor

# In[153]:


from xgboost.sklearn import XGBRegressor


# In[154]:


xg_reg = XGBRegressor() 


# In[155]:


xg_reg.fit(X_train, y_train) 


# In[156]:


# Predicting the values using the model


# In[157]:


y_pred_xg1 = xg_reg.predict(X_train) 


# In[158]:


y_pred_xg2 = xg_reg.predict(X_test) 


# In[159]:


# plot predicted data 
plt.plot(y_test,y_pred_xg2,"bo")  
plt.title('XG bossting') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[160]:


# RMSE values


# In[161]:


print('RMSE Training data:', np.sqrt(metrics.mean_squared_error(y_train,y_pred_xg1)))


# In[162]:


print('RMSE Testing data:', np.sqrt(metrics.mean_squared_error(y_test,y_pred_xg2)))


# ### Hyper parameter tuning for XG Boosting

# In[163]:


import xgboost


# In[164]:


regressor=xgboost.XGBRegressor()


# In[165]:


booster=['gbtree']
base_score=[0.5]


# In[166]:


n_estimators = [100, 200]
max_depth = [3, 4]
learning_rate=[0.1,0.2]


# In[167]:


# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'booster':booster,
    'base_score':base_score
    }


# In[168]:


from sklearn.model_selection import RandomizedSearchCV


# In[175]:


# Set up the random search with 3-fold cross validation
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=3, n_iter=20,
            scoring = 'neg_root_mean_squared_error',n_jobs = 3,
            verbose = 5, 
            return_train_score = True,
            random_state=36)


# In[176]:


random_cv.fit(X_train,y_train)


# In[177]:


random_cv.best_estimator_


# In[178]:


regressor = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=4,
             min_child_weight=1,  monotone_constraints='()',
             n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)


# In[179]:


regressor.fit(X_train,y_train)


# In[180]:


# Predicting the output values


# In[181]:


y_pred_xgb1 = regressor.predict(X_train)


# In[182]:


y_pred_xgb2 = regressor.predict(X_test)


# In[183]:


# plot predicted data 
plt.plot(y_test,y_pred_xgb2,"bo")  
plt.title('XG bossting') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[184]:


# RMSE values


# In[185]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_train,y_pred_xgb1)))


# In[186]:


print('RMSE of testing data:', np.sqrt(metrics.mean_squared_error(y_test,y_pred_xgb2)))


# #### "Random Forest Regressor" with RMSE of 0.16 is the best model

# ### Creating the pickle file of the model

# In[188]:


import pickle
with open('model_rf_inventory.pkl','wb') as f:
    pickle.dump(regressor,f)


# In[190]:


# Load from file
with open('model_rf_inventory.pkl', 'rb') as file:
    pickle_model = pickle.load(file)


# In[191]:


# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X_test)


# In[ ]:




