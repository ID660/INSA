#!/usr/bin/env python
# coding: utf-8

# # Import module and create sparksession

# In[3]:


from pyspark.sql.types import *
from pyspark.sql.functions import* # function pour calculer somme etc
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover
# sparksession
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Sentiment Analysis Spark').config("spark.some.config.option", "some-value").getOrCreate()


# # Read datafile into SparkDataframe

# In[8]:


# read ccsv file into dataframe with autatically inferredschema
create_csv = spark.read.csv(r'C:\v_data\tweets.csv', inferSchema =True, header =True)
create_csv.show(truncate=False, n=3)


# # Selected the related data

# In[16]:



data = create_csv.select("SentimentText", col("Sentiment").cast("Int").alias("label"))
data.show(truncate = False,n=5)


# # divide data into training and testing
# 

# In[19]:


#divide data, 70% for training, 30% for testing
divideData= data.randomSplit([0.7, 0.3])
trainingData= divideData[0]#index 0 = data training
testingData =divideData[1]  #index 1 = data testing
train_rows = trainingData.count()
test_rows= testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)


# # Prepare training Data

# In[ ]:


#Separate "SentimentText" into individual words using tokenizer


# In[20]:


tokenizer = Tokenizer(inputCol="SentimentText", outputCol="SentimentWords")
tokenizedTrain= tokenizer.transform(trainingData)
tokenizedTrain.show(truncate=False,n =3)#Show top 5 rows and full column contents (PySpark)


# In[ ]:


# removing stopwords(unimportant words to be features)


# In[23]:


swr= StopWordsRemover(inputCol = tokenizer.getOutputCol(), outputCol="MeaningFullWords")
swrRemovedTrain= swr.transform(tokenizedTrain)
swrRemovedTrain.show(truncate=False, n= 3)


# In[27]:


#Converting words feature into numerical feature.
#In Spark 2.2.1,it is implemented in HashingTF funtion using Austin Appleby's MurmurHash 3 algorithm
hashTF= HashingTF(inputCol=swr.getOutputCol(), outputCol='features')
numericTrainingData= hashTF.transform(swrRemovedTrain).select('label', 'MeaningFullWords', 'features')
numericTrainingData.show(truncate=False, n =3 )
#hashTF = HashingTF(inputCol=swr.getOutputCol(), outputCol="features")
#numericTrainData = hashTF.transform(SwRemovedTrain).select(
#    'label', 'MeaningfulWords', 'features')
#numericTrainData.show(truncate=False, n=3)


# # train our classifier model using traindata with  model logistic regression

# In[28]:


lr= LogisticRegression(labelCol='label', featuresCol='features', maxIter=10, regParam=0.01)
model= lr.fit(numericTrainingData)
print("training is OK")


# # prepare testing Data

# In[29]:


tokenizedTest = tokenizer.transform(testingData)
tokenizedTest.show(truncate=False, n=5)


# In[30]:


swrRemovedTest= swr.transform(tokenizedTest) # unimportant words


# In[31]:


numericTestData= hashTF.transform(swrRemovedTest).select('label', 'MeaningFullWords', 'features') # wordsintonumeric
numericTestData.show(truncate=False, n=4)


# # Predict testing data and calculate the accuracy model

# In[43]:


import pyspark.ml
##val denseVector = r.getAs[org.apache.spark.ml.linalg.SparseVector]("features").toDense
#org.apache.spark.mllib.linalg.Vectors.fromML(denseVector)
#import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
#import org.apache.spark.mllib.linalg.Vectors


# In[48]:


get_ipython().system(' pip install spark-1.4.1-bin-hadoop2.6/bin/spark-submit --driver-memory 5g --packages com.databricks:spark-csv_2.10:1.1.0 fbi_spark.py')


# In[ ]:


prediction = model.predict(numericTestData)
predictionFinal = prediction.select('label', 'MeaningFullWords', 'features')
predictionFinal.show(truncate=False, n=5)
correctPrediction = predictionFinal.filter(
    predictionFinal['prediction'] == predictionFinal['Label']).count()
totalData = predictionFinal.count()
print("correct prediction:", correctPrediction, ", total data:", totalData, 
      ", accuracy:", correctPrediction/totalData)

