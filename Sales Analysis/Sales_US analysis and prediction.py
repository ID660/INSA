#!/usr/bin/env python
# coding: utf-8

# # Découverte de la description des variables

# In[2]:


import pandas as pd
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


files = [ file for file in os.listdir(r'C:\Sales_US')]
for file in files:
        print(file)                            
    


# In[5]:


import pandas as pd
path = r'C:\Sales_US'
big_data = (pd.DataFrame())
for file in files:
    current_data= pd.read_csv(path+'/'+file)
    big_data= pd.concat([big_data, current_data])
print(big_data)


# In[6]:


big_data.head()


# In[7]:


big_data.info()


# In[8]:


big_data.describe()


# In[9]:


big_data.describe(include='all')


# In[11]:


big_data.shape


# In[12]:


donnee_janvier = pd.read_csv(path+'/'+'/Sales_January_2019.csv')
donnee_janvier.shape


# In[ ]:


# transformation en un fichier csv


# In[13]:


big_data.to_csv(path+'/big_data.csv', index=False)


# In[14]:


big_data.dtypes


# In[17]:


big_data.isna().sum()  # 545 missing values


# In[ ]:


# les valeurs manquantes et suppression
#Drop the rows where at least one element is missing:
#df.dropna()
#Drop the columns where at least one element is missing:df.dropna(axis='columns')
#Drop the rows where all elements are missing: dropna(how=all)


# In[16]:


big_data = big_data.dropna(how='all')
big_data.shape


# In[ ]:


#2 step


# In[18]:


big_data.head()


# # le mois durant lequel le meilleur chiffre d'affaire est réalisé?
# 

# In[19]:


def month(x):
    return x.split('/')[0]


# In[20]:


big_data['Month']= big_data['Order Date'].apply(month)


# In[21]:


big_data


# In[22]:


big_data['Month'].unique()


# In[23]:


big_data = big_data[big_data['Month']!='Order Date']


# In[24]:


big_data['Month'].unique()


# In[25]:


big_data.dtypes


# In[26]:


big_data['Month']= big_data['Month'].astype(int)


# In[27]:


big_data.dtypes


# In[28]:


big_data['Price Each']= big_data['Price Each'].astype(float)
big_data['Quantity Ordered']= big_data['Quantity Ordered'].astype(int)


# In[30]:


big_data.dtypes


# In[32]:


big_data['sales']= big_data['Quantity Ordered'] *big_data['Price Each']  


# In[33]:


big_data.head()


# In[35]:


# grouper par mois la somme des ventes(sales)
big_data.groupby('Month')['sales'].sum()


# In[38]:


month= range(1,13)
plt.bar(month, big_data.groupby('Month')['sales'].sum())
plt.xticks(month)
plt.ylabel('sales in USD')
plt.xlabel('month')
plt.show()


# In[41]:


# histogramme de ventes( sales) par mois

big_data.hist(column='sales',by='Month')


# # Dans quelle ville nous avons enregistre  un maximum de commmande?

# In[43]:


'682 Chestnut St, Boston, MA 02215'.split(',')[1]


# In[1]:


def city(x):
    return x.split(',')[1]


# In[2]:


big_data['city']= big_data['Purchase Address'].apply(city)


# In[46]:


big_data


# In[6]:


import pandas as pd
df = pd.read_csv('big_data.csv')
df.head()


# In[7]:


big_data = df


# In[8]:


big_data = big_data.dropna(how='all')
big_data.shape
#df.dropna(inplace = True)  supprime toutes les lignes NAN


# In[9]:


def month(x):
    return x.split('/')[0]


# In[10]:


big_data['Month']= big_data['Order Date'].apply(month)


# In[28]:


big_data2 = big_data.copy()


# In[29]:


big_data2.head()


# In[12]:


big_data 


# In[ ]:


big_data['Month']= big_data['Month'].astype(int)


# In[14]:


big_data.dtypes


# In[33]:


big_data = big_data.astype({"Month":'int'}) 
#df = df.astype({"Name":'category', "Age":'int64'}) 

