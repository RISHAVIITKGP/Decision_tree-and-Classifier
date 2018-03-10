
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[32]:


from sklearn import tree


# In[6]:


df = pd.read_csv('train_new.csv')


# In[7]:


df.describe()


# In[8]:


df_x = df.drop(['X_Minimum','X_Maximum','Y_Maximum','Y_Minimum','Faults'],axis = 1)


# In[9]:


df_x = df_x.to_dict(orient='report')


# In[10]:


from sklearn.feature_extraction import DictVectorizer


# In[11]:


Dict_vec = DictVectorizer()


# In[13]:


df_x = Dict_vec.fit_transform(df_x).toarray()


# In[15]:


df_y = df[['Faults']]


# In[16]:


df_y = np.asarray(df_y)


# In[17]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


le = LabelEncoder()


# In[21]:


le.fit(df_y)


# In[22]:


df_y = le.transform(df_y)


# In[23]:


df_y


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train,X_test,Y_train,Y_test = train_test_split(df_x,df_y,test_size=0.3,random_state = 40)


# In[27]:


X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])


# In[33]:


model = tree.DecisionTreeClassifier(criterion = 'gini')


# In[34]:


model.fit(X_train,Y_train)


# In[37]:


model.score(X_test,Y_test)


# In[36]:


model.score(X_train,Y_train)


# In[39]:


predicted = model.predict(X_test)


# In[41]:


from sklearn.metrics import confusion_matrix


# In[42]:


confusion_matrix(predicted,Y_test)

