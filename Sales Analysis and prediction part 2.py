#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


files = [ file for file in os.listdir(r'C:\Sales')]
for file in files:
        print(file)                            
    


# In[3]:


import pandas as pd
path = r'C:\Sales'
big_data = (pd.DataFrame())
for file in files:
    current_data= pd.read_csv(path+'/'+file)
    big_data= pd.concat([big_data, current_data])
print(big_data)


# In[4]:


# no feedback
big_data.to_csv(path+'/big_data.csv', index=False)


# In[5]:


big_data.isna().sum()


# In[6]:


big_data = big_data.dropna(how='all')
big_data.shape


# In[7]:


def month(x):
    return x.split('/')[0]


# In[8]:


big_data['Month']= big_data['Order Date'].apply(month)


# In[9]:


big_data['Month'].unique()


# In[10]:


big_data = big_data[big_data['Month']!='Order Date']


# In[11]:


big_data['Month'].unique()


# In[12]:


big_data.dtypes


# In[13]:


big_data['Month']= big_data['Month'].astype(int)


# In[14]:


big_data.dtypes


# In[15]:


big_data['Price Each']= big_data['Price Each'].astype(float)
big_data['Quantity Ordered']= big_data['Quantity Ordered'].astype(int)


# In[16]:


big_data['sals']= big_data['Quantity Ordered'] *big_data['Price Each']  


# In[17]:


big_data.head()


# In[19]:


#grouper par mois la somme des ventes(sales)
big_data.groupby('Month')['sals'].sum()


# In[21]:


month= range(1,13)
plt.bar(month, big_data.groupby('Month')['sals'].sum())
plt.xticks(month)
plt.ylabel('sales in USD')
plt.xlabel('month')
plt.show()


# #  Les Villes  qui recoivent le plus de commande?

# In[22]:


def city(x):
    return x.split(',')[1]


# In[23]:


big_data['city']= big_data['Purchase Address'].apply(city)


# In[24]:


big_data


# In[25]:


# compter le nombre de commande par ville
big_data.groupby('city')['city'].count()


# In[32]:


big_data.groupby('city')['city'].count().index


# In[33]:


big_data.groupby('city')['city'].count().values


# In[34]:



plt.bar(big_data.groupby('city')['city'].count().index, big_data.groupby('city')['city'].count().values)
plt.xticks(rotation='vertical')
plt.ylabel('receive orders')
plt.xlabel('city name')
plt.show()


# In[29]:


#grouper par produit  les villes
big_data.groupby('city')['Product'].count()


# In[36]:


plt.bar(big_data.groupby('city')['Product'].count().index, big_data.groupby('city')['Product'].count().values)
plt.xticks(rotation='vertical')
plt.ylabel('receive orders')
plt.xlabel('product')
plt.show()


# In[ ]:


# ville qui recoivent le plus de commande San francisco suivie de Los angeles
# La ville d'Atlanta et Austiin recoivent le moins de commande à savoir pourquoi..


# # A qu'elle moment devrait on faire une campagne publicitaire pour plus de vente?

# In[38]:


big_data['Hour']= pd.to_datetime(big_data['Order Date']).dt.hour


# In[39]:


big_data


# In[40]:


keys=[]
hours=[]
for key,hour in big_data.groupby('Hour'):
    keys.append(key)# pour chaque boucle, le nombre de fois que la boucle se repete append(ajouter)
    hours.append(len(hour))
hours# nombre de commande par heure
# exple8 a 3910 fois


# In[41]:


keys


# In[42]:


big_data.groupby('Hour')


# In[44]:


plt.grid()
plt.plot(keys, hours)
plt.xlabel('heure de la journée keys')
plt.ylabel('nombre de commandes hours')


# In[ ]:


# meilleur créneau pour lancer campagne publicitaire entre  environ vers10h  et et vers 17h


# # Quel produit se vend le plus?

# In[45]:


big_data.groupby('Product')['Quantity Ordered'].sum()


# In[47]:


big_data.groupby('Product')['Quantity Ordered'].sum().plot(kind='bar')


# In[ ]:


les produits qui se vendent le plus sont les batteries etc..


# In[ ]:


#L'impact du prix


# In[48]:


# 
big_data.groupby('Product')['Price Each'].sum().plot(kind='bar')


# In[49]:


big_data.groupby('Product')['Price Each'].mean()


# In[50]:


big_data.groupby('Product')['Price Each'].mean().plot(kind='bar')


# In[53]:


# grouper par produit les quantites en index
products= big_data.groupby('Product')['Quantity Ordered'].sum().index
# grouper par produit la somme des quantites
quantity=  big_data.groupby('Product')['Quantity Ordered'].sum()
prices = big_data.groupby('Product')['Price Each'].mean()


# In[63]:


plt.figure(figsize=(40, 24))
fig,ax1= plt.subplots()
ax2= ax1.twinx()
ax1.bar(products, quantity, color='g')
ax2.plot(products, prices, color='g')
ax1.set_xticklabels(products, rotation='vertical', size=8)


# In[ ]:


Les produits vendus à grande quantité sont à petits pris et les produits qui sont vensus à petite quantité sont à trés frand prix..


# # Combinaisons de produits qui se vendent le plus?

# In[64]:


big_data.to_csv(path+'/big_datav.csv', index=False)


# In[1]:


import pandas as pd
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('big_data.csv')
df.head()
big_data


# In[11]:


big_datav = pd.read_csv('big_datav.csv') 
big_datav


# In[13]:


df = big_datav[big_datav['Order ID'].duplicated(keep=False)]
df


# In[14]:


# prendre les combinaisons qui ont été commandé en même temps
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x:','.join(x))


# In[15]:


df


# In[17]:


df2 = df.drop_duplicates(subset=['Order ID'])
df2


# In[20]:


df2.Grouped.value_counts()[0:5].plot.pie()


# #  Application des time series pour une analyse plus fine

# In[41]:


b = pd.read_csv('big_datav.csv')
b.head()


# In[44]:


b['Order Date'] = pd.to_datetime(b['Order Date'], errors = 'coerce')
b.head()


# In[46]:


# On rajoute le jour de la semaine
b['jour_semaine'].dt.weekday
b


# In[50]:


b['day'] =  b['Order Date'].dt.day
b['Order_date'] = b['Order Date'].dt.strftime('%d/%m/%y')
b['year'] = b['Order Date'].dt.year
b


# In[52]:



b = b.drop(['year', 'Order Date', 'new_formatted_date'], axis =1)


# In[53]:


b


# In[55]:


path = r'C:\Sales'
b.to_csv(path+'/b.csv', index=False)


# In[56]:


b.to_csv('b.csv', index= False)


# In[57]:


data_date= pd.read_csv('b.csv', index_col = 'Order_date', parse_dates = True)
data_date.head()


# In[58]:


data_date.index


# In[59]:


b.dtypes


# In[66]:


#big_date['index_col'] = big_date['index_col'].apply( lambda x: x[1:11])
data_date['jour_semaine'] = data_date['jour_semaine'].astype('object')
data_date['Month'] = data_date['Month'].astype('object')
data_date['Order ID'] = data_date['Order ID'].astype('object')
data_date['Hour'] = data_date['Hour'].astype('object')
data_date['day'] = data_date['day'].astype('object')


# In[67]:


data_date.dtypes


# In[68]:


data_date


# In[77]:


date= pd.read_csv('b.csv', index_col = 'Order_date', parse_dates = True)
date.head()


# In[80]:


date.loc['2019','Quantity Ordered'].resample('M').plot()
plt.show()


# In[93]:


date.loc['2019','sals'].resample('M').mean().plot()
plt.show()


# In[94]:


date.loc['2019','sals'].resample('W').mean().plot()
plt.show()
# volatilité prix vente dans la semaine


# In[81]:


date.loc['2019','Quantity Ordered'].resample('M').mean().plot()
plt.show()


# In[82]:


date.loc['2019','Quantity Ordered'].resample('W').mean().plot()
plt.show()
##L'analyse de la  volatilité du quantitié commandé en terme de moyenne, min et max selon les date


# In[89]:


sem = date.loc['2019','Quantity Ordered'].resample('W').agg(['mean', 'std','min','max'])
plt.figure(figsize= (12, 8))
sem['mean']['2019'].plot(label= 'moyenne la semaine ')
plt.fill_between(sem.index, sem['max'], sem['min'], alpha = 0.2, label= 'min-max week order quantity')
plt.legend()
plt.show()


# In[83]:


date.loc['2019','Quantity Ordered'].resample('W').agg(['mean', 'std','min','max'])


# In[ ]:


# transformation des données


# In[96]:


#big_data['Month'].unique()
date['Product'].unique()


# In[98]:


date =  date.drop(['day', 'Purchase Address', 'Order ID'], axis =1)


# In[99]:


date.head()


# In[100]:


date['city'] = data_date['city'].astype('object')
date['Month'] = data_date['Month'].astype('object')
date['Product'] = data_date['Product'].astype('object')
date['Hour'] = data_date['Hour'].astype('object')
date['jour_semaine'] = data_date['day'].astype('object')


# In[101]:


date.dtypes


# In[102]:


# Create DataFrame 
df = pd.DataFrame(date) 

for _c in df.select_dtypes(include=['object']).columns:
    print(_c)
    df[_c]  = pd.Categorical(df[_c])
business = pd.get_dummies(df)
business


# In[125]:


business.dtypes


# In[106]:


get_ipython().system(' pip install statsmodels')


# In[113]:


from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
#Fit the model
mod = sm.tsa.statespace.SARIMAX(df['Price Each'], trend='c', order=(1,1,4), seasonal_order=(0,0,0,12))
res = mod.fit(disp=False)
print(res.summary())


# In[114]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
pred = res.predict(185950, 185961)
bc_pred = pd.concat([df['Price Each'], pred])
plt.plot(bc_pred)
plt.axvline(x='2020-01-01', color='red')


# In[116]:


get_ipython().system(' pip install prophet')


# In[118]:


from prophet import Prophet


# In[130]:


c = pd.read_csv('big_datav.csv')
c.head()


# In[ ]:


c1 = c[['Price Each', 'Month']]


# In[139]:


get_ipython().system('pip install pycaret pandas shap')


# In[141]:


get_ipython().system(' pip install pycaret')


# In[142]:


import pandas as pd
from pycaret.classification import *


# In[ ]:


#np.zeros((130164, 106659), dtype='uint8')
cat_features = ['Quantity Ordered', 'Month', 'city']
experiment = setup(c, target='sals', categorical_features=cat_features)

