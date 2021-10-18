#!/usr/bin/env python
# coding: utf-8

# # Task 1: Project Overview and Import Modules

# In[1]:


#%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))


# # Task 2: Load and Explore the NER Dataset

# In[ ]:


#La colonne "Sentence #" représente l'identifiant de la phrase.
#La colonne Word représente un des mots de la phrase.
#La colonne POS donne la nature grammatical de notre mot.
#La colonne Tag correspond à l'entité que l'on recherchera à prédire. pour appliquer une première mise sous forme des données.


# In[2]:


df = pd.read_csv("ner_dataset.csv", encoding="latin1")
df = df.fillna(method="ffill") # filling null values
df.head()


# In[3]:


df = df.drop('POS',  1)
df = df.groupby('Sentence #').agg(list)
df = df.reset_index(drop=True)
df.head()
#Essential info about tagged entities:

#geo = Geographical Entity
#org = Organization
#per = Person
#gpe = Geopolitical Entity
#tim = Time indicator
#art = Artifact
#eve = Event
#nat = Natural Phenomenon


# In[ ]:


##Afficher quelques observations des données mises en forme (prêt pour le modèle).


# In[4]:


#Looking for null values
print(df.isnull().sum())


# In[5]:


import pandas as pd
df1 = pd.read_csv("ner_dataset.csv", encoding="latin1")
df1 = df1.fillna(method="ffill") # filling null values
#df.head()
#Filling Null Values
df1.head()


# In[6]:


print("Unique Words in corpus:",df1['Word'].nunique())
print("Unique Tag in corpus:",df1['Tag'].nunique())


# In[7]:


#Storing unique words and tags as a list
words = list(set(df1['Word'].values))
words.append("ENDPAD")
num_words = len(words)


# In[8]:


tags = list(set(df1['Tag'].values))
num_tags = len(tags)


# In[9]:


num_words, num_tags


# # Task 3: Retrieve Sentences and Corresponsing Tags

# In[10]:


#Creating a class to get data in desired formate. i.e. Word,POS,Tag
class SentanceGetter(object):
    def __init__(self,df1):
        self.n_sent = 1 #counter
        self.df1 = df1
        agg_func = lambda s:[(w,p,t) for w,p,t in zip(s['Word'].tolist(),s['POS'].tolist(),s['Tag'].tolist())]
        self.grouped = self.df1.groupby("Sentence #").apply(agg_func)
        self.sentances = [s for s in self.grouped]


# In[11]:


getter = SentanceGetter(df1)
sentances = getter.sentances


# In[12]:


sentances[0]


# In[ ]:


#Définir un modèle permettant de résoudre ce problème et afficher son résumé.


# # Task 4: Define Mappings between Sentences and Tags

# In[13]:


word2idx =  {w : i+1 for i,w in enumerate(words)}#and the same applies for the named entities but we need to map our labels to numbers this time:
tag2idx  =  {t : i for i,t in enumerate(tags)}
#enumerate nous permet d'itérer à travers une séquence mais il garde une trace à la fois de l'index et de l'élément


# In[14]:


word2idx


# # Task 5: Padding Input Sentences and Creating Train/Test Splits

# In[15]:


plt.hist([len(s) for s in sentances], bins=50)
plt.show()


# In[ ]:


'''from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_len = 50

X = [[word2idx[w[0]] for w in s]for s in sentances]
X = pad_sequences(maxlen = max_len , sequences =X, padding='post', value =num_words-1)

y = [[tag2idx[w[2]] for w in s]for s in sentances]
y = pad_sequences(maxlen = max_len , sequences =y, padding='post', value =tag2idx["O"])
y = [to_categorical(i, num_classes=num_tags) for i in y]
y = [to_categorical(i, num_classes=num_tags) for i in y]'''


# In[ ]:


#Justifier la fonction de perte qui sera utilisée pour entraîner ce modèle.


# In[17]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 50
X = [[word2idx[w[0]] for w in s] for s in sentances]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)
y = [[tag2idx[w[2]] for w in s] for s in sentances]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[19]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # Task 6: Build and Compile a Bidirectional LSTM Model

# In[20]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[21]:


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()
 


# In[ ]:


#the summary shows that we have 1.88 million of parameters to be trained.
#Now let’s compile our model and specify the loss function, the matrix we want to track, and the optimizer function. We’ll use adam optimizer here, sparce_categorical_crossentropy
#as the loss function and the matrix we gonna concern is accuracy matrix.


# In[22]:


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# # Task 7: Train the Model

# In[23]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback


# In[ ]:


#Then all we have left to start training is to call model.fit() 
#, and we’ll pass our training data which is x_train and y_train. 
#Then we’ll create our validation data by further splitting training data.
#You can increase the batch_size if you have GPU of more memory size. 
#Here we will use just 3 epochs as it takes more than 10 to 15 minutes to train the model if we use more epochs


# In[24]:


get_ipython().run_cell_magic('time', '', 'chkpt = ModelCheckpoint("model_weights.h5", monitor=\'val_loss\',verbose=1, save_best_only=True, save_weights_only=True, mode=\'min\')\nearly_stopping = EarlyStopping(monitor=\'val_accuracy\', min_delta=0, patience=2, verbose=0, mode=\'max\', baseline=None, restore_best_weights=False)\ncallbacks = [PlotLossesCallback(), chkpt, early_stopping]\nhistory = model.fit(\n    x=x_train,\n    y=y_train,\n    validation_data=(x_test,y_test),\n    batch_size=32, \n    epochs=3,\n    callbacks=callbacks,\n    verbose=1\n)')


# In[ ]:


#We can see at the bottom right end that the accuracy of our model is more than 98% .


# # Task 8: Evaluate Named Entity Recognition Mode

# In[26]:


model.evaluate(x_test, y_test)


# # PREDICTION

# In[27]:


i = np.random.randint(0, x_test.shape[0]) #659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

