#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


df = pd.read_csv(r'D:\dataset\listings.csv')


# In[48]:


df


# In[49]:


df = df.sort_values(by='reviews_per_month' , ascending=False)


# In[50]:


df


# In[51]:


df.head()


# In[52]:


df.shape


# In[53]:


df.columns


# In[89]:


sns.distplot(df['review_scores_location'])


# In[90]:


sns.boxplot(df['reviews_per_month'])


# In[91]:


sns.distplot(df['reviews_per_month'])


# In[54]:


df[df['instant_bookable'].apply(lambda instant_bookable : instant_bookable[0] == 't')]


# In[55]:


dict1 = {'t' : True , 'f' : False}


# In[58]:


df['instant_bookable'].map(dict1)


# In[60]:


df.groupby(['instant_bookable'])['review_scores_communication' , 'review_scores_value' , 'review_scores_location' ].describe()


# In[62]:


df.groupby(['instant_bookable'])['review_scores_communication' , 'review_scores_value' , 'review_scores_location' ].agg([np.mean , np.max])


# In[65]:


sns.lmplot(x = 'review_scores_communication' , y = 'review_scores_location' , data = df , hue = 'instant_bookable' , fit_reg = False)


# In[66]:


_ , axes = plt.subplots(nrows = 1 , ncols = 2 , sharey = True , figsize = (10,7))
sns.boxplot(x = 'instant_bookable' , y = 'review_scores_location' , data = df , ax = axes[0])
sns.violinplot(x = 'instant_bookable' , y = 'review_scores_communication' , data = df , ax = axes[1])


# In[69]:


sns.countplot(x = 'review_scores_value' , hue = 'instant_bookable' , data = df )


# In[23]:


pd.crosstab(df['review_scores_communication'] , df['reviews_per_month'] , normalize=True)


# In[27]:


plt.scatter(df['calculated_host_listings_count'] , df['reviews_per_month'])


# In[28]:


sns.jointplot(x='calculated_host_listings_count' , y = 'reviews_per_month' , data = df , kind = 'scatter')


# In[30]:


df.head()


# In[77]:


sns.jointplot(x='review_scores_value' , y = 'reviews_per_month' , data = df , kind = 'kde')


# In[ ]:





# In[78]:


df.head()


# In[79]:


features = ['review_scores_communication' , 'review_scores_location']


# In[82]:


df[features].hist(figsize=(10,7))


# In[83]:


df[features].plot(kind='density' , subplots=True , layout=(1,2))


# In[ ]:





# In[88]:


plt .figure(figsize=(10,8))
sns.heatmap(df.corr() , annot=True)


# In[92]:


df


# In[94]:


_ , axes = plt.subplots(nrows = 1 , ncols = 2 , sharey = True , figsize = (10,7))
sns.boxplot(x = 'instant_bookable' , y = 'review_scores_communication' , data = df , ax = axes[0])
sns.violinplot(x = 'instant_bookable' , y = 'review_scores_value' , data = df , ax = axes[1])


# In[ ]:




