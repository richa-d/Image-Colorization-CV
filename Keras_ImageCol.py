
# coding: utf-8

# In[3]:


from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from PIL import Image


# In[ ]:

# TO RESIZE TRAIN IMAGES TO 400 X 400
# img_list = []
for root, dirs, files in os.walk('/scratch/rsd352/proj_cv/data/train'):
     for file in files:
#         print(os.path.join(root, file))
        if file[-3:] == 'jpg':
            im_path = os.path.join(root, file)
#             print(im_path)
            img = Image.open(im_path)
            img = img.resize((400, 400), Image.ANTIALIAS)
          #   img_list.append(img)
            img.save(im_path) 
print(len(img_list))


# # In[22]:


# # print(img_list[0].size)


# # In[49]:

# TO RESIZE VAL IMAGES TO 400 X 400
# vimg_list = []
for root, dirs, files in os.walk('/scratch/rsd352/proj_cv/data/val'):
     for file in files:
#         print(os.path.join(root, file))
        if file[-3:] == 'jpg':
            im_path = os.path.join(root, file)
#             print(im_path)
            img = Image.open(im_path)
            img = img.resize((400, 400), Image.ANTIALIAS)
            vimg_list.append(img)
            img.save(im_path) 
print(len(vimg_list))


# In[4]:


new_list = []
for root, dirs, files in os.walk('/scratch/rsd352/proj_cv/data/train'):
     for file in files:
#         print(os.path.join(root, file))
        if file[-3:] == 'jpg':
            im_path = os.path.join(root, file)
          #   print(im_path)
            image = img_to_array(load_img(im_path))
            image = np.array(image, dtype=float)
            new_list.append(image)
# print(len(new_list))

print("Train loaded")
# In[5]:


# print(new_list[0].shape)


# In[50]:


vnew_list = []
for root, dirs, files in os.walk('/scratch/rsd352/proj_cv/data/val'):
     for file in files:
#         print(os.path.join(root, file))
        if file[-3:] == 'jpg':
            im_path = os.path.join(root, file)
          #   print(im_path)
            image = img_to_array(load_img(im_path))
            image = np.array(image, dtype=float)
            vnew_list.append(image)
# print(len(vnew_list))


# In[6]:

print("Val loaded")


# X = np.empty((3, 400, 400, 1))
# Y = np.empty((3, 400, 400, 2))
Xl = []
Yl = []
for image in new_list:
# for i in range(10):
    # image = new_list[i]
    a = rgb2lab(1.0/255*image)[:,:,0]
#     print(X.shape)
    b = rgb2lab(1.0/255*image)[:,:,1:]
    b /= 128
    a = a.reshape(400, 400, 1)
#     print(a)
    b = b.reshape(400, 400, 2)
    Xl.append(a)
    Yl.append(b)
#     print(X)
#     X = np.append(X, a)
#     print(X)
#     Y = np.append(Y, b)
# print(X.shape)
X = np.asarray(Xl)
# print(X.shape)
# print(Y.shape)
Y = np.asarray(Yl)
# print(Y.shape)

print("Train transformed")

# In[51]:


# X = np.empty((3, 400, 400, 1))
# Y = np.empty((3, 400, 400, 2))
vXl = []
vYl = []
for image in new_list:
# for i in range(10):
    # image = vnew_list[i]
    a = rgb2lab(1.0/255*image)[:,:,0]
#     print(X.shape)
    b = rgb2lab(1.0/255*image)[:,:,1:]
    b /= 128
    a = a.reshape(400, 400, 1)
#     print(a)
    b = b.reshape(400, 400, 2)
    vXl.append(a)
    vYl.append(b)
#     print(X)
#     X = np.append(X, a)
#     print(X)
#     Y = np.append(Y, b)
# print(X.shape)
vX = np.asarray(vXl)
# print(vX.shape)
# print(Y.shape)
vY = np.asarray(vYl)
# print(vY.shape)

print("Val transformed")


# In[21]:


# print(X[0].shape)
# print(Y[0].shape)
# print(X[0])


# In[56]:


# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))


# In[57]:


model.compile(optimizer='rmsprop', loss='mse')


# In[58]:


# Xs = X[7].reshape(1, 400, 400, 1)
# Ys = Y[7].reshape(1, 400, 400, 2)


# In[59]:


model.fit(x=X, y=Y, batch_size=1, epochs=10)


# In[62]:


# vXs = vX[5].reshape(1, 400, 400, 1)
# vYs = vY[5].reshape(1, 400, 400, 2)

# print(model.evaluate(Xs, Ys, batch_size=1))
# print(model.evaluate(vXs, vYs, batch_size=1))
# output = model.predict(vXs)
print(model.evaluate(vX, vY, batch_size=1))
output = model.predict(vX)
output *= 128
# print(output.shape)
# Output colorizations
# cur = np.zeros((400, 400, 3))
# cur[:,:,0] = vXs[0][:,:,0]
# cur[:,:,1:] = output[0]
# imsave("/Users/richadeshmukh/Desktop/trial.png", lab2rgb(cur))
# imsave("/Users/richadeshmukh/Desktop/trial_bw.png", rgb2gray(lab2rgb(cur)))


# In[66]:


i = 0
for op in output:
#     print(op.shape)
    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = vX[i][:,:,0]
    cur[:,:,1:] = op
    imsave("/scratch/rsd352/proj_cv/data/img_output/img"+str(i)+".png", lab2rgb(cur))
    imsave("/scratch/rsd352/proj_cv/data/img_output/img-bw"+str(i)+".png", rgb2gray(lab2rgb(cur)))
    i += 1

