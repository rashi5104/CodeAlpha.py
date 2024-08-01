#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[3]:


df=pd.read_csv(r"C:\\Users\\rashi\\Dropbox\\PC\\Downloads\\archive (11)\\NFLX.csv")


# In[4]:


df


# In[5]:


df.head()


# In[7]:


df.shape


# In[9]:


# Visualize the close price data
plt.figure(figsize=(16,8))
plt.title('Netflix')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.show()


# In[10]:


df=df[['Close']]
df.head()


# In[11]:


# Create a variable to predict 'x' days out into the future
future_days = 25
# Create a new column (target) shifted 'x' units/days up
df['Prediction']= df[['Close']].shift(-future_days)
df.tail(4)


# In[12]:


# Create the feature dataset (X) and convert it to a numpy array and remove the last 'x'  rows/days
X=np.array(df.drop(['Prediction'],1))[:-future_days]
print(X)


# In[15]:


# Create the target dataset (Y) and convert it to anumpy array and get all of the target values except the last  'x'rows/days
y=np.array(df['Prediction'])[:-future_days]
print(y)


# In[19]:


#Splitting the data into 75% training and 25% testing
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[20]:


# Create the models
# Create the decision tree regression model
tree=DecisionTreeRegressor().fit(x_train,y_train)
# Create the linear regression model
lr= LinearRegression().fit(x_train,y_train)


# In[21]:


# Get the last 'x' rows of the feature dataset
x_future=df.drop(['Prediction'],1)[:-future_days]
x_future=x_future.tail(future_days)
x_future=np.array(x_future)
x_future


# In[23]:


# Show the model tree prediction
tree_prediction= tree.predict(x_future)
print(tree_prediction)
print()
# Show the model linear regression prediction
lr_prediction= lr.predict(x_future)
print(lr_prediction)


# In[24]:


# Visualize the data
predictions= tree_prediction
valid=df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[25]:


predictions= lr_prediction
valid=df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[ ]:




