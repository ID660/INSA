#!/usr/bin/env python
# coding: utf-8

# In[31]:


import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np


# In[32]:


#Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)


# In[33]:


model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')# movenet and multipose
movenet = model.signatures['serving_default']# setting variable


# In[34]:


from cv2 import VideoCapture
from cv2 import waitKey


# In[ ]:


import cv2

import sys


cpt=0

vidStream=cv2.VideoCapture(0)
while True:
    ret,frame=vidStream.read()

    cv2.imshow("test frame",frame)
    cv2.imwrite(r"C:\v_data\train.jpg"%cpt,frame)
    cpt +=1

    if cv2.waitKey(10)==ord('q'):
        break


# In[36]:


cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read() #returns ret and the frame
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


#Lorsque j'utilise cv2.waitKey(1),je reçois un flux vidéo en direct continu de la webcam de mon ordinateur portable. 
#Cependant, lorsque j'utilise cv2.waitKey(0), j'obtiens des images fixes. Chaque fois que je ferme la fenêtre, une autre apparaît avec une autre photo prise à l'époque. Pourquoi ne s'affiche-t-il pas en continu ?


# In[37]:


cap = cv2.VideoCapture(0)# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):  # to exit the frame
        break
cap.release()
cv2.destroyAllWindows()


# In[38]:


frame


# In[39]:


plt.imshow(frame)


# In[40]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# In[41]:


cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read() #returns ret and the frame
       # Resize image
      img = frame.copy()
      img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
      input_img = tf.cast(img, dtype=tf.int32)
      # Detection section
      results = movenet(input_img)
      print(results)  
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


y, x, score* 17


# In[42]:


results


# In[43]:


results.keys()


# In[44]:


results['output_0'].numpy()[ : , : , :51]# 51 values represnet the keypoints


# In[ ]:


cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read() #returns ret and the frame
       # Resize image
      img = frame.copy()
      img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
      input_img = tf.cast(img, dtype=tf.int32)
      # Detection section
      results = movenet(input_img)
      print(results)  
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


cap = cv2.VideoCapture(0)# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(keypoints_with_scores)
    
    # Render keypoints 
    #loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[45]:


cap = cv2.VideoCapture(0)# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(keypoints_with_scores)
    
    # Render keypoints 
    #loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[46]:


plt.imshow( keypoints_with_scores)


# In[47]:


keypoints_with_scores


# In[48]:


keypoints_with_scores[0]# first person, high value correspond to higher body low value with lower part of body exple leg


# In[51]:


#Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


# In[63]:


cap = cv2.VideoCapture(0)# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    print(keypoints_with_scores)
    
    # Render keypoints 
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)
    
    cv2.imshow('movement', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[64]:


cap = cv2.VideoCapture('novak.mp4')# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints 
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[61]:


frame.shape


# In[60]:


plt.imshow(frame)


# # 3.Draw keypoints

# In[49]:


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


# # 4.Draw Edges

# In[53]:


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# In[55]:


def draw_connections(frame, keypoints, edges, confidence_threshold):# draw (dessiner connexions)
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


# In[ ]:


frame


# In[ ]:


plt.imshow(frame)


# In[ ]:


cap = cv2.VideoCapture(r"C:\v_data\novak.mp4')# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


cap = cv2.VideoCapture('novak.mp4')# establish connection
while cap.isOpened():
    ret, frame = cap.read() # return value(ret) frame ( is the image you want)
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints 
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
    
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


cv2.namedWindow("lll")
cap = cv2.VideoCapture(1)
while( cap.isOpened() ) :
    ret,img = cap.read()
    cv2.imshow("lll",img)
    k = cv2.waitKey(10)
    if k == 12:
        break


# In[ ]:


#Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
   for person in keypoints_with_scores:
       draw_connections(frame, person, edges, confidence_threshold)
       draw_keypoints(frame, person, confidence_threshold)


# In[ ]:


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


# In[ ]:


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# In[ ]:


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

