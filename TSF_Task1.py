#!/usr/bin/env python
# coding: utf-8

# ## # <U>Data Science & Business AnalyticsTasks</u>

# TASK 1 - Predict the percentage of an student based on the no. of study hours.

# Name: Priyanka Kadam

#  Dataset Link : http://bit.ly/w-data

# What will be predicted score if a student studies for 9.25 hrs/ day?

# IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# IMPORTING DATASET AND Read Dataset

# In[2]:


data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data.shape


# In[3]:


data.head(5)


# In[4]:


data.tail(5)


# In[5]:


data.describe()


# In[6]:


data.info()


# In[23]:


data.isnull().sum()


# In[ ]:





# In[7]:


data.plot(x="Hours", y="Scores",style="o")
plt.title("Distribution of Scores")
plt.xlabel("Study Hours")
plt.ylabel("Obtained Score")
plt.show()


# By observing the graph we can say there is a strong linear relationship between scores and study hours as it resmbles a straight line. So this dataset is ideal to perform linear regression.

# TRAIN AND TEST DATA SPLIT

# In[8]:


x1 = data.iloc[:,0].values
y1 = data.iloc[:,1].values
x = x1.reshape(-1,1)
y = y1.reshape(-1,1)


# RANDOM LINEAR REGRESSION MODEL

# Applying on TEST data

# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression
linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)


# In[11]:


line = linearRegressor.coef_*x+linearRegressor.intercept_
plt.scatter(x,y)
plt.plot(x,line, color="pink");
plt.show()


# ACCURACY SCORE FROM TRAINING AND TEST DATA

# In[12]:


print('Test Score')
print(linearRegressor.score(x_test, y_test))
print('Training Score')
print(linearRegressor.score(x_train, y_train))


# In[13]:


print(x_test) # Testing data - In Hours
y_pred = linearRegressor.predict(x_test)


# In[14]:



df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1


# PREDICTION ON TEST AND TRAIN DATA

# In[19]:


y_pred= linearRegressor.predict(x_test)
x_pred= linearRegressor.predict(x_train)


# SOLLUTION OF PROBLEM STATEMENT

# In[20]:


print('Score of student who studied for 9.25 hours a date', linearRegressor.predict([[9.25]]))


# SOLLUTION OF PROBLEM STATEMENT

# In[21]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


# In[22]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))


# AFTER BUILDING THE LINEAR REGRESSION MODEL AND OBTAIN THE PREDICTION WE CAN CONCLUDE "PREDICTED SCORE OF A STUDENT STUDIES WHO STUDY 9.25 HRS/ DAY WILL BE ABLE TO SCORE 93.69173249"

# In[ ]:





# In[ ]:





# In[ ]:




