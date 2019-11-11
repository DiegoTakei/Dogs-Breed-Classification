#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as img
import PIL.Image


# In[ ]:


train = np.load('../dog-breed-identification/train.zip')


# In[2]:


import glob
import random
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from IPython.display import HTML

def get_thumbnail(path): 
    image = img.imread(path)
    return image

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


# In[ ]:


dogs = pd.read_csv('../dog-breed-identification/labels.csv')
#dogs = dogs.sample(50)
dogs['file'] = dogs.id.map(lambda id: f'/home/diegotakei/workspace/dog/dog-breed-identification/train/{id}.jpg')
dogs['image'] = dogs.file.map(lambda f: get_thumbnail(f))
dogs.head()


# In[3]:


dogs = pd.read_csv('../dogs_data/labels.csv')
dogs = dogs[:5000]
dogs['file'] = dogs.id.map(lambda id: f'../dogs_data/resized/{id}.jpg')


# In[4]:


imgs = []
for f in dogs['file'].values:
    imgs.append(get_thumbnail(f))
imgs = np.array(imgs)

imgs.shape


# In[ ]:


imgs


# In[ ]:


int_row_to_float = lambda row: list(map(lambda element: float(element), row))
int_matrix_to_float = lambda matrix: list(map(lambda row: int_row_to_float(row), matrix))
img_to_float = lambda img: list(map(lambda dimension: int_matrix_to_float(dimension), img))

float_imgs = list(map(lambda img: img_to_float(img), imgs[:5]))
float_imgs


# In[5]:


float_imgs = imgs.astype(float)
float_imgs.shape


# In[ ]:


dogs['Indexes'] = dogs["breed"].str.startswith('appenzeller')
    
result = dogs[(dogs.Indexes == True)]
result.tail()

pd.set_option('display.max_colwidth', -1)
HTML(result[['breed', 'image']].to_html(formatters={'image': image_formatter}, escape=False))
#pd.reset_option('all')


# In[6]:


train_images = float_imgs

train_labels = dogs['breed']


# In[10]:


def create_network(features, labels, mode):
    i = tf.reshape(features['x'], [-1, 256, 256, 3])
    
    # receives [batch_size, 256, 256, 3]
    # returns [batch_size, 256, 256, 32]
    convolution1 = tf.layers.conv2d(inputs = i, filters = 32, kernel_size = [5,5], activation = tf.nn.relu,
                                 padding = 'same')
    
    # receives [batch_size, 256, 256, 3]
    # returns [batch_size, 128, 128, 32]
    pooling1 = tf.layers.max_pooling2d(inputs = convolution1, pool_size = [2,2], strides = 2)
    
    # receives [batch_size, 128, 128, 32]
    # returns [batch_size, 128, 128, 64]
    convolution2 = tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size = [5,5], activation = tf.nn.relu,
                                  padding = 'same')
    
    # receives [batch_size, 128, 128, 64]
    # returns [batch_size, 64, 64, 64]
    pooling2 = tf.layers.max_pooling2d(inputs = convolution2, pool_size = [2,2], strides = 2)
    
    # receives [batch_size, 128, 128, 64]
    # returns [batch_size, 128, 128, 128]
    convolution3 = tf.layers.conv2d(inputs = pooling2, filters = 32, kernel_size = [5,5], activation = tf.nn.relu,
                                 padding = 'same')
    
    # receives [batch_size, 7, 7, 64]
    # returns [batch_size, 3136]
    flattening = tf.reshape(pooling2, [-1, 7 * 7 * 64])
    
    # 3136 inputs -> 1024 neurons on hidden layer -> 10 outputs
    # receives [batch_size, 3136]
    # returns [batch_size, 1024]
    dense = tf.layers.dense(inputs = flattening, units = 1024, activation = tf.nn.relu)
    
    dense2 = tf.layers.dense(inputs = dense, units = 1024, activation = tf.nn.relu)
    
    # dropout
    dropout =  tf.layers.dropout(inputs = dense2, rate = 0.2, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    # receives [batch_size, 1024]
    # returns [batch_size, 10]
    output = tf.layers.dense(inputs = dropout, units = 10)
    
    predictions = tf.argmax(output, axis = 1)
    
    if(mode == tf.estimator.ModeKeys.PREDICT):
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)  
    
    losses = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = output)
    
    if(mode == tf.estimator.ModeKeys.TRAIN):
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train = optimizer.minimize(losses, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = losses, train_op = train)
    
    if(mode == tf.estimator.ModeKeys.EVAL):
        eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = predictions)}
        return tf.estimator.EstimatorSpec(mode = mode, loss = losses, eval_metric_ops = eval_metrics_ops) 

classifier = tf.estimator.Estimator(model_fn = create_network)

train_function = tf.estimator.inputs.numpy_input_fn(x = {'x': train_images}, y = train_labels, 
                                                        batch_size= 10000, num_epochs= None, shuffle= True)
classifier.train(input_fn = train_function, steps = 2000)


# In[ ]:





# In[ ]:





# In[ ]:




