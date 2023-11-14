#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.simplefilter("ignore")


# In[6]:


import pandas as pd
df = pd.read_csv("avocado.csv")


# In[7]:


df


# In[8]:


df.dropna()


# In[11]:


df.drop(["Unnamed: 0", "Date", "region"], axis = 1, inplace = True)


# In[12]:


df


# In[13]:


from sklearn.preprocessing import LabelEncoder
df["type"] = LabelEncoder().fit_transform(df["type"])
df


# In[14]:


features = len(df.columns)
y = df.iloc[:, 0:1].values
X = df.iloc[:, 1:features].values


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size = 0.2)
X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size = 0.5)


# In[23]:


from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors = 8).fit(X_train, y_train)
model.score(X_test, y_test)


# In[24]:


scores = []
results = 0
best_score = 0
neighbors = range(1,15)

for i in neighbors:
    knn = KNeighborsRegressor(n_neighbors = i).fit(X_train, y_train)
    results = knn.score(X_test, y_test)
    scores.append(round(results,2))
    
    if results > best_score:
        best_score = results
        best_k = i 
        bestmodel = knn
print(scores)
print(best_k)


# In[25]:


import matplotlib.pyplot as plt
plt.plot(neighbors, scores)


# In[27]:


accuracy = bestmodel.score(X_validate, y_validate)
print("The best model has an accuracy of:", round(accuracy, 2))


# In[30]:


#Linear Regression
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression().fit(X_train, y_train)
score = modelLR.score(X_test, y_test)
print(score)


# In[32]:


#Decsion Tree Regression
from sklearn.ensemble import RandomForestRegressor
modelRF = RandomForestRegressor(criterion = "mse", max_leaf_nodes = 100).fit(X_train, y_train.ravel())
score = modelRF.score(X_test, y_test)
y_pred = modelRF.predict(X_test)
print(score)


# In[34]:


from sklearn.metrics import mean_squared_error
print("Accuracy:", score)
print("MSE:" + str(mean_squared_error(y_test, y_pred)))


# In[ ]:




