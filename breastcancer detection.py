#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df = pd.read_csv("breast-cancer.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.head(5)


# In[6]:


df.info()


# In[7]:


#return all the columns with null values count
df.isna().sum()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df['diagnosis']


# In[11]:


df['diagnosis'].value_counts()


# In[12]:



# Assuming 'diagnosis' column contains categorical values
df['diagnosis'] = df['diagnosis'].astype('category')

# Create countplot
sns.countplot(data=df, x='diagnosis')


# In[13]:



df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})


# In[14]:


df.head()


# In[15]:



columns = df.columns[1:5]


pd.plotting.scatter_matrix(df[columns], figsize=(10, 10))


plt.show()


# In[16]:


sns.pairplot(df.iloc[:,1:5],hue = "diagnosis")


# In[29]:


df.iloc[:,1:32].corr()


# In[26]:




plt.figure(figsize=(10, 10))
sns.heatmap(df.iloc[:, 1:10].corr(), annot=True, fmt=".0%")


# In[45]:


X = df.iloc[:,2:31].values
Y = df.iloc[:, 1].values


# In[47]:


print(X)


# In[48]:


print(Y)


# In[51]:


#spliting data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)


# In[52]:


#feature scaling
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# In[53]:


X_train


# In[55]:


X_test


# In[62]:


def models(X_train, Y_train):
    # Logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    # Decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=0, criterion="entropy")
    tree.fit(X_train, Y_train)
    
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(random_state=0, criterion="entropy", n_estimators=10)
    forest.fit(X_train, Y_train)
    
    print('[0] Logistic regression accuracy:', log.score(X_train, Y_train))
    print('[1] Decision tree accuracy:', tree.score(X_train, Y_train))
    print('[2] Random Forest accuracy:', forest.score(X_train, Y_train))
    
    return log, tree, forest


# In[63]:


model=models(X_train,Y_train)


# In[65]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy',accuracy_score(Y_test,model[i].predict(X_test)))


# In[74]:


pred = model[0].predict(X_test)
print('Predicted Values:')
print(pred)
print('Actual values:')
print(Y_test)


# In[77]:


pred = model[1].predict(X_test)
print('Predicted Values:')
print(pred)
print('Actual values:')
print(Y_test)


# In[78]:


pred = model[2].predict(X_test)
print('Predicted Values:')
print(pred)
print('Actual values:')
print(Y_test)


# In[79]:


from joblib import dump
dump(model[2],"Cancer_prediction.joblib")


# In[ ]:




