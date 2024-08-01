#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data Collection & Processing

# In[3]:


titanic_data=pd.read_csv(r"C:\\Users\\rashi\\Dropbox\\PC\\Downloads\\train.csv")


# In[4]:


titanic_data


# In[5]:


titanic_data.head()


# In[6]:


titanic_data.shape


# In[7]:


titanic_data.info()


# In[8]:


titanic_data.isnull().sum()


# # Handling the missing values

# In[10]:


# drop the cabin column from the dataset
titanic_data=titanic_data.drop(columns='Cabin',axis=1)


# In[11]:


titanic_data.info()


# In[12]:


# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[13]:


# finding the mode value of embarked column
print(titanic_data['Embarked'].mode())


# In[14]:


print(titanic_data['Embarked'].mode()[0])


# In[16]:


# replacing the missing value in embarked column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[17]:


titanic_data.isnull().sum()


# # Data Analysis

# In[18]:


# getting some statistical measures about the data
titanic_data.describe()


# In[21]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# # Data Visualization

# In[20]:


sns.set()


# In[22]:


# making a countplot for survived column
sns.countplot('Survived',data=titanic_data)


# In[24]:


sns.countplot('Sex',data=titanic_data)


# In[25]:


titanic_data['Sex'].value_counts()


# In[28]:


# number of survived Gender wise
sns.countplot('Sex',hue='Survived',data=titanic_data)


# In[29]:


# making a countplot for pclass column
sns.countplot('Pclass',data=titanic_data)


# In[30]:


sns.countplot("Pclass",hue='Survived',data=titanic_data)


# # Encoding the categorical columns

# In[31]:


titanic_data['Sex'].value_counts()


# In[32]:


titanic_data['Embarked'].value_counts()


# In[36]:


# converting categorical columns 
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[37]:


titanic_data.head()


# # Seperating features & Target

# In[39]:


X=titanic_data.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
Y=titanic_data['Survived']


# In[40]:


print(X)


# In[41]:


print(Y)


# # Splitting the data into training data & test data

# In[42]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[43]:


print(X.shape,X_train.shape,X_test.shape)     


# # Model Training
# # Logistic Regression

# In[44]:


model=LogisticRegression()


# In[45]:


# training the logistic regresssion model with training data
model.fit(X_train,Y_train)


# # Model Evaluation

# In[50]:


# accuracy on training data
X_train_prediction=model.predict(X_train)


# In[51]:


print(X_train_prediction)


# In[53]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of training data : ' , training_data_accuracy)


# In[54]:


X_test_prediction=model.predict(X_test)


# In[55]:


print(X_test_prediction)


# In[56]:


test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print('Accuracy score of test data :',test_data_accuracy)


# In[ ]:




