#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\\Users\\rashi\\Dropbox\\PC\\Downloads\\archive (12)\\train.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


for column in df.columns:
    print(df[column].value_counts())
    print("*"*20)


# In[8]:


df.drop(columns=['lot_size','lot_size_units'],inplace=True)


# In[9]:


df.describe()


# In[10]:


df['beds'].value_counts()


# In[11]:


df.head()


# # Price per sq feet

# In[12]:


df['price_per_sqft']= df['price']*10000/df['size']


# In[13]:


df['price_per_sqft']


# In[14]:


df.describe()


# In[15]:


df.shape


# In[16]:


df


# In[17]:


df.drop(columns=['size_units'],inplace=True)


# In[18]:


df.drop(columns=['price_per_sqft'],inplace=True)


# In[19]:


df.head()


# In[20]:


df.to_csv("final_dataset_csv")


# In[21]:


X=df.drop(columns=['price'])
y=df['price']


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[24]:


print(X_train.shape)
print(y_train.shape)


# # Applying Linear Regression

# In[25]:


column_trns=make_column_transformer((OneHotEncoder(sparse=False),['beds']),remainder='passthrough')


# In[26]:


scaler=StandardScaler()


# In[27]:


lr=LinearRegression(normalize=True)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


# x being feature metrics
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
lr=LinearRegression()
lr.fit(X_scaled,y)


# In[32]:


pipe=make_pipeline(column_trns,scaler,lr)


# In[33]:


pipe.fit(X_train,y_train)


# In[34]:


y_pred_lr=pipe.predict(X_test)


# In[35]:


r2_score(y_test,y_pred_lr)


# # using lasso

# In[36]:


lasso=Lasso()


# In[37]:


pipe=make_pipeline(column_trns,scaler,lasso)


# In[38]:


pipe.fit(X_train,y_train)


# In[39]:


y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)


# # using ridge

# In[40]:


ridge=Ridge()


# In[41]:


pipe=make_pipeline(column_trns,scaler,ridge)


# In[42]:


pipe.fit(X_train,y_train)


# In[43]:


y_pred_ridge=pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)


# In[44]:


print("No Regularization: " , r2_score(y_test,y_pred_lr))
print("Lasso:" , r2_score(y_test,y_pred_lasso))
print("Ridge:" , r2_score(y_test,y_pred_ridge))


# In[45]:


import pickle


# In[46]:


pickle.dump(pipe,open('RidgeModel.pkl','wb'))


# In[ ]:




