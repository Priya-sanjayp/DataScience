#!/usr/bin/env python
# coding: utf-8

# # Python Assignment - 2 - Google Playstore Analysis

# ### Importing Libraries and Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('playstore_analysis.csv',encoding="ISO-8859–1")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# # Tasks

# ## 1. Data clean up – Missing value treatment

# ### a. Drop records where rating is missing since rating is our target/study variable

# In[8]:


df.isnull().sum()


# In[9]:


# Since there are 1474 record with null values/missing values we delete those records, since rating is our target/study variable
df.dropna(subset=['Rating'], axis=0, inplace=True)


# In[10]:


df.isnull().sum()


# In[11]:


df


# ### b. Check the null values for the Android Ver column.

# In[12]:


df.isnull().sum()


# In[13]:


# There are only 3 null values in Android Ver Column


# #### i. Are all 3 records having the same problem?

# In[14]:


nullvalues=df.loc[df['Android Ver'].isnull()]


# In[15]:


nullvalues


# #### ii. Drop the 3rd record i.e. record for “Life Made WI-Fi Touchscreen Photo Frame”

# In[16]:


df.drop([10472],inplace=True)


# In[17]:


nullvalues


# In[18]:


df


# #### iii. Replace remaining missing values with the mode

# In[19]:


# mode of Android Ver column
df['Android Ver'].mode()


# In[20]:


df['Android Ver'].fillna(df['Android Ver']. mode()[0], inplace=True)


# In[21]:


df.isnull().sum()


# As you can see above there are no null values remaining in Android Ver column

# ### c. Current ver – replace with most common value

# In[22]:


# mode of Current Ver column
df['Current Ver'].mode()


# In[23]:


df['Current Ver'].fillna(df['Current Ver']. mode()[0], inplace=True)


# In[24]:


df.isnull().sum()


# As you can see above there are no null values remaining in any of the column.

# ## 2. Data clean up – correcting the data types

# ### a. Which all variables need to be brought to numeric types?

# In[25]:


df.dtypes


# In[26]:


df.head()


# Reviews, Installs and Price these are the columns which must be changed into numeric types.

# ### b. Price variable – remove $ sign and convert to float

# In[27]:


df["Price"] = df["Price"].map(lambda x: x.lstrip("$"))
df["Price"] = df["Price"].astype(float)


# In[28]:


df.info()


# ### c. Installs – remove ‘,’ and ‘+’ sign, convert to integer

# In[29]:


df["Installs"] = df["Installs"].map(lambda x: x.replace(",",""))
df["Installs"] = df["Installs"].map(lambda x: x.rstrip("+"))
df["Installs"] = df["Installs"].astype('int')


# In[30]:


df.info()


# ### d. Convert all other identified columns to numeric

# In[31]:


df["Reviews"] = df["Reviews"].astype('int')


# In[32]:


df.info()


# ## 3. Sanity checks – check for the following and handle accordingly

# ### a. Avg. rating should be between 1 and 5, as only these values are allowed on the play store.

# #### i. Are there any such records? Drop if so.

# In[33]:


df.loc[df.Rating<1] & df.loc[df.Rating>5]


# There are no such records.

# ### b. Reviews should not be more than installs as only those who installed can review the app.

# #### i. Are there any such records? Drop if so.

# In[34]:


df.drop(df[df["Reviews"]>df["Installs"]].index, inplace = True)


# In[35]:


df.info()


# ## 4. Identify and handle outliers –

# ### a. Price column

# #### i. Make suitable plot to identify outliers in price

# In[36]:


plt.boxplot(df['Price'])
plt.title('Outlies')
plt.ylabel('Price')
plt.show()


# There are many outliers as you an see.

# #### ii. Do you expect apps on the play store to cost $200? Check out these cases

# In[37]:


print('Yes, There are apps on the play store which cost $200')
df[df["Price"]>200]


# #### iii. After dropping the useless records, make the suitable plot again to identify outliers

# In[38]:


df.drop(df[df["Price"]>200].index, inplace = True )
plt.boxplot(x=df["Price"])
plt.title('Outlies')
plt.ylabel('Price')
plt.show()


# There are many outliers as you an see. Hence we can not drop them.

# #### iv. Limit data to records with price < $30

# In[39]:


replace_30 = df[df['Price'] > 30].index
df.drop(labels=replace_30, inplace=True)


# In[40]:


count = df.loc[df['Price'] > 30].index
count.value_counts().sum()


# ### b. Reviews column

# #### i. Make suitable plot

# In[41]:


sns.distplot(df['Reviews'])
plt.title('Reviews')
plt.show()


# #### ii. Limit data to apps with < 1 Million reviews

# In[42]:


ld = df[df['Reviews'] > 1000000 ].index
df.drop(labels = ld, inplace=True)
print(ld.value_counts().sum(),'columns dropped')


# In[43]:


df.shape


# ### c. Installs

# #### i. What is the 95th percentile of the installs?

# In[44]:


df['Installs'].describe


# In[45]:


percentile = df.Installs.quantile(0.95)


# In[46]:


print(percentile,"is 95th percentile of Installs")


# #### ii. Drop records having a value more than the 95th percentile

# In[47]:


df.drop(df[df["Installs"]>10000000.0].index, inplace = True )


# In[48]:


df.shape


# # Data analysis to answer business questions

# ## 5. What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower values?

# ### a. How do you explain this?

# In[49]:


sns.distplot(df['Rating'],color ='red')
plt.title('Distribution Plot')
print('The skewness is',df['Rating'].skew())
print('The Median of this distribution {} is greater than mean {}, Hence it is left-Skewed Distribution'.format(df.Rating.median(),df.Rating.mean()))
plt.show()


# ### b. What is the implication of this on your analysis?

# In[50]:


df['Rating'].mode()


# Since median > mean, the distribution left Skewed.The distribution of Rating is more Skewed towards lower values. Most of data is distributed on the right side of the graph i.e., between the range of 4-5 which shows most of the apps.

# ## 6. What are the top Content Rating values?

# ### a. Are there any values with very few records?

# In[51]:


df['Content Rating'].value_counts()


# This fields "Unrated" and "Adults only 18+" have very few records.

# ### b. If yes, drop those as they won’t help in the analysis

# In[52]:


few = (df.groupby('Content Rating').filter(lambda x : len(x)<=3)).index
df.drop(few, inplace= True)


# In[53]:


df['Content Rating'].value_counts()


# ## 7. Effect of size on rating

# ### a. Make a joinplot to understand the effect of size on rating

# In[54]:


sns.jointplot(x ='Rating', y ='Size', data = df, kind ='kde')
plt.ylabel('Size')
plt.xlabel('Rating')
plt.show()


# ### b. Do you see any patterns?

# Yes, patterns can be observed between Size and Rating. They are in corelation. The apps having rating between 4-5 are more satured around size of 20000 i.e. 20 MB

# ### c. How do you explain the pattern?

# Rating and size are proportional to each other as size increase rating also increases.
# GBut this is not always true i.e. for higher Rating, their is constant Size. Thus we can conclude that there is positive corelation between Size and Rating.

# ## 8. Effect of price on rating

# ### a. Make a jointplot (with regression line)

# In[57]:


sns.jointplot(x='Rating', y='Price', data=df, kind='reg')
plt.show()


# 

# ### b. What pattern do you see?

# I was not able to see clear pattern pattern, but on increasing the Price, Rating remains almost constant greater than 4.

# ### c. How do you explain the pattern?

# Since on increasing the Price, Rating remains almost constant greater than 4. Thus it can be concluded that there is very weak Positive correlation between Rating and Price.

# ### d. Replot the data, this time with only records with price > 0

# In[59]:


df1=df.loc[df.Price>0]
sns.jointplot(x='Rating',y='Price', data=df1, kind='reg')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.show()


# ### e. Does the pattern change?

# Yes, On limiting the record with Price > 0, the overall pattern changed, their is very weakly Negative Correlation between Price and Rating

# ### f. What is your overall inference on the effect of price on the rating

# Generally increasing the Prices, doesn't have signifcant effect on Higher Rating. For Higher Price, Rating is High and almost constant ie greater than 4

# ## 9. Look at all the numeric interactions together –

# ### a. Make a pairplort with the colulmns - 'Reviews', 'Size', 'Rating', 'Price'

# In[63]:


sns.pairplot(df, vars=['Reviews', 'Size', 'Rating', 'Price'], kind='reg')
plt.show()


# ## 10. Rating vs. content rating

# ### a. Make a bar plot displaying the rating for each content rating

# In[64]:


df.groupby(['Content Rating'])['Rating'].count().plot.bar(color="orange")
plt.show()


# ### b. Which metric would you use? Mean? Median? Some other quantile?

# In[65]:


sns.boxplot(df['Rating'])
plt.show()


# We must use Median in this case as we are having Outliers in Rating. Because in case of Outliers , median is the best measure of central tendency

# ### c. Choose the right metric and plot

# In[66]:


df.groupby(['Content Rating'])['Rating'].median().plot.barh(color="red")
plt.show()


# ## 11. Content rating vs. size vs. rating – 3 variables at a time

# ### a. Create 5 buckets (20% records in each) based on Size

# In[68]:


bins=[0, 20000, 40000, 60000, 80000, 100000]
df['Bucket Size'] = pd.cut(df['Size'], bins, labels=['0-20','20-40','40-60','60-80','80-100'])
pd.pivot_table(df, values='Rating', index='Bucket Size', columns='Content Rating')


# ### b. By Content Rating vs. Size buckets, get the rating (20th percentile) for each combination

# In[69]:


ab=pd.pivot_table(df, values='Rating', index='Bucket Size', columns='Content Rating', aggfunc=lambda x:np.quantile(x,0.2))
ab


# ### c. Make a heatmap of this

# #### i. Annotated

# In[72]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(ab, annot=True, linewidths=.5, cmap='Blues', fmt='.1f',ax=ax)
plt.show()


# #### ii. Greens color map

# In[73]:


f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(ab, annot=True, linewidths=.5, cmap='Greens',fmt='.1f',ax=ax)
plt.show()


# ### d. What’s your inference? Are lighter apps preferred in all categories? Heavier? Some?

# Based on the analysis, it is not true that smaller apps are preferred in all categories. Because apps with size 40-60 MB and 80-100 MB have got the highest rating in all cateegories. So, in general we can conclude that bigger apps are preferred in all categories

# In[ ]:




