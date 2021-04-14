#!/usr/bin/env python
# coding: utf-8

# # Python Assignment 1 - E-Commerce – EDA

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')


# ## Loading DataSet

# In[153]:


# Importing Dataset, specified encoding to remove errors  
df = pd.read_csv('C:/Users/akshay/Desktop/Board Infinity/Python/Python Assignment 1/Ecommerce.csv',encoding = 'ISO-8859-1')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# ## 1. Perform Basic EDA

# There are only two Numeric variables Quantity, UnitPrice and CustomerID

# ### a. Boxplot – All Numeric Variables

# In[7]:


plt.subplots(figsize=(10,7))
df.boxplot(['UnitPrice','Quantity'],grid=True, color='red')
plt.show()


# ### b. Histogram – All Numeric Variables

# In[8]:


plt.subplots(figsize=(10,7))
plt.hist(df['Quantity'],color='orange')
plt.hist(df['UnitPrice'],color='green')
plt.show()


# ### c. Distribution Plot – All Numeric Variables

# In[9]:


plt.subplots(figsize=(10,7))

sns.set_style('whitegrid')
sns.distplot(df['CustomerID'],kde=False, color ='blue' , bins =30)
plt.title('Customer ID Distribution')
plt.show()


# In[10]:


plt.subplots(figsize=(10,7))

sns.set_style('whitegrid')
sns.distplot(df['Quantity'],kde=False, color ='green' , bins =30)
plt.title('Quantity Distribution')
plt.show()


# In[11]:


plt.subplots(figsize=(10,7))

sns.set_style('whitegrid')
sns.distplot(df['UnitPrice'],kde=False, color ='red' , bins =30)
plt.title('Unit Price Distribution')
plt.show()


# ### d. Aggregation for all numerical Columns

# In[12]:


df.describe()


# In[13]:


# our data we have some invoices with 0 unit price, so I have calculated aggregation again excluding 0 unit price/\. 
df = df[df['UnitPrice'] > 0]
df.describe()


# ### e. Unique Values across all columns

# In[14]:


# used this command to find unique values for all columns in dataset
df.nunique()


# ### f. Duplicate values across all columns

# In[15]:


df.duplicated()


# In[16]:


## Count of all duplicate values across columns
print("Duplicate values across all columns: ",df.duplicated().sum(),'\n','\n')


print("Duplicate Values in column InvoiceNo: ",df['InvoiceNo'].duplicated().sum())
print("Duplicate Values in column StockCode: ",df['StockCode'].duplicated().sum())
print("Duplicate Values in column Description: ",df['Description'].duplicated().sum())
print("Duplicate Values in column Quantity: ",df['Quantity'].duplicated().sum())
print("Duplicate Values in column InvoiceDate: ",df['InvoiceDate'].duplicated().sum())
print("Duplicate Values in column UnitPrice: ",df['UnitPrice'].duplicated().sum())
print("Duplicate Values in column CustomerID: ",df['CustomerID'].duplicated().sum())
print("Duplicate Values in column Country: ",df['Country'].duplicated().sum())


# ### g. Correlation – Heatmap - All Numeric Variables

# In[17]:


sns.heatmap(df.corr(),annot=True)
plt.title('Correlation Heatmap')
plt.show()


# ### h. Regression Plot - All Numeric Variables

# In[18]:


sns.lmplot(x='UnitPrice', y='Quantity', data=df, hue='Country')
plt.title('Regression Plot')
plt.show()


# ### i. Bar Plot – Every Categorical Variable vs every Numerical Variable

# In[ ]:





# ### j. Pair plot - All Numeric Variables

# In[159]:


sns.pairplot(df, size=3, palette='dark')
plt.title('Pairplot')
plt.show()


# ### k. Line chart to show the trend of data - All Numeric/Date Variables

# In[ ]:





# ### l. Plot the skewness - All Numeric Variables

# In[154]:


df.skew(axis = 0, skipna = True)


# In[155]:


df.skew(axis = 1, skipna = True)


# In[156]:


from scipy.stats import skew   
import pylab as p


# In[157]:


x1 = np.linspace( -5, 5, 1000 ) 
y1 = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 
  
p.plot(x1, y1, '*') 
  
print( '\nSkewness for data : ', skew(y1))


# In[158]:


x1 = np.linspace( -5, 12, 1000 ) 
y1 = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 
  
p.plot(x1, y1, '.') 
  
print( '\nSkewness for data : ', skew(y1))


# ## 2. Check for missing values in all columns and replace them with the appropriate metric
# (Mean/Median/Mode)

# In[39]:


# check missing values for each column
df.isnull().sum()


# #### we can not replace missing values in since the values assigned to those variables are unique.

# In[79]:


# df without missing values
df = df.dropna()


# In[80]:


df.isnull().sum()


# In[81]:


df


# ## 3. Remove duplicate rows

# In[82]:


df


# In[83]:


df.duplicated()


# In[84]:


## Count of all duplicate values across columns
print("Duplicate values across all columns: ",df.duplicated().sum())


# In[85]:


df=df.drop_duplicates()
df


# ## 4. Remove rows which have negative values in Quantity column

# In[86]:


df


# In[87]:


df = df[df.Quantity > 0]
df


# ## 5. Add the columns - Month, Day and Hour for the invoice

# In[88]:


df.info()


# In[89]:


# we need to change the format of InvoiceDate to Date Time Format
df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
df.info()


# In[90]:


df['Month']=df['InvoiceDate'].dt.month
df['Day']=df['InvoiceDate'].dt.day
df['Hour']=df['InvoiceDate'].dt.time


# In[91]:


df.head()


# ## 6. How many orders made by the customers?

# In[92]:


df.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count().head()


# In[93]:


Orders = df.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()

plt.subplots(figsize=(15,6))
plt.plot(Orders.CustomerID, Orders.InvoiceNo)
plt.xlabel('Customers ID')
plt.ylabel('Number of Orders')
plt.title('Number of Orders made by Customers')
plt.show()


# ## 7. TOP 5 customers with higher number of orders

# In[95]:


print('The TOP 5 Customers with Higher number number of orders...')
Orders.sort_values(by='InvoiceNo', ascending=False).head()


# ##  8. How much money spent by the customers?

# In[96]:


# Since we dont know the amount spent by the customer we have to calculate it
df['Money_Spent'] = df['Quantity'] * df['UnitPrice']


# In[97]:


df.head()


# In[98]:


amount_spent = df.groupby(by=['CustomerID','Country'], as_index=False)['Money_Spent'].sum()
amount_spent.head()


# In[99]:


plt.subplots(figsize=(15,6))
plt.plot(amount_spent.CustomerID, amount_spent.Money_Spent)
plt.xlabel('Customers ID')
plt.ylabel('Money spent')
plt.title('Money Spent by the Customers')
plt.show()


# ## 9. TOP 5 customers with highest money spent

# In[100]:


print('The TOP 5 customers with highest money spent...')
amount_spent.sort_values(by='Money_Spent', ascending=False).head()


# ## 10. How many orders per month?

# In[128]:


df.groupby('InvoiceNo')['Month'].unique().value_counts().iloc[:-1].sort_index()


# In[131]:


ax = df.groupby('InvoiceNo')['Month'].unique().value_counts().iloc[:-1].sort_index().plot(kind='bar',color='orange',figsize=(15,6))
ax.set_xlabel('Month',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders by Month',fontsize=15)
ax.set_xticklabels(('Feb','March','April','May','June','July','August','September','October','November','Deccember'), rotation='horizontal', fontsize=15)
plt.show()


# ## 11. How many orders per day?

# In[123]:


df.groupby('InvoiceNo')['Day'].unique().value_counts().iloc[:-1].sort_index()


# In[127]:


ax = df.groupby('InvoiceNo')['Day'].unique().value_counts().iloc[:-1].sort_index().plot(kind='bar',color='orange',figsize=(15,6))
ax.set_xlabel('Day',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders by Day',fontsize=15)
ax.set_xticklabels((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30), rotation='horizontal', fontsize=15)
plt.show()


# ## 12. How many orders per hour?

# In[115]:


# The Hour time is in minutes and seconds as ewell, hence we convert it to hour only
df['Exact Hour'] = pd.DatetimeIndex(df['InvoiceDate']).hour
df.head()


# In[124]:


df.groupby('InvoiceNo')['Exact Hour'].unique().value_counts().iloc[:-1].sort_index()


# In[125]:


ax = df.groupby('InvoiceNo')['Exact Hour'].unique().value_counts().iloc[:-1].sort_index().plot(kind='bar',color='orange',figsize=(15,6))
ax.set_xlabel('Hour',fontsize=15)
ax.set_ylabel('Number of Orders',fontsize=15)
ax.set_title('Number of orders by Hours',fontsize=15)
ax.set_xticklabels(range(6,21), rotation='horizontal', fontsize=15)
plt.show()


# ## 13. How many orders for each country?

# In[132]:


country_orders = df.groupby('Country')['InvoiceNo'].count().sort_values()


# In[133]:


country_orders


# In[134]:


country_orders.plot.bar()
plt.title('Orders for each Country')
plt.show()


# ## 14. Orders trend across months

# In[138]:


orders_month = df.groupby('Month')['InvoiceNo'].count().sort_values()
orders_month


# In[147]:


def show_plot(orders_month,
             figsize=(15,15),
             color='blue',
             linestyle='-',
             xlabel='Month',
             ylabel='Orders',
             label='Orders Trend'):
        plt.figure(figsize=figsize)
        plt.plot(orders_month.index,orders_month,color=color,label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc=2)


# In[148]:


show_plot(orders_month,label='All data')


# ## 15. How much money spent by each country?

# In[152]:


country_amount_spent = df.groupby('Country')['Money_Spent'].sum().sort_values()

plt.subplots(figsize=(15,8))
country_amount_spent.plot(kind='bar', fontsize=12, color='Orange')
plt.xlabel('Money Spent (Dollar)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.title('Money Spent by Each Country', fontsize=12)
plt.show()

