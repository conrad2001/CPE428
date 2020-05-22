import keras
from keras.models import Model
from keras.layers import *

import numpy as np
import cv2 as cv
import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.model_selection import train_test_split



# To make sure the code runs the same way every time
# , I set the seed for the Numpy and Tensorflow random number generators.
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)


# This downloads and extracts the AT&T Faces Datset.

faces,labels = fetch_olivetti_faces(return_X_y=True)
faces = np.reshape(faces,(-1,64,64,1))

# Print some info about the dataset:
print('Faces shape:',faces.shape)
print('Labels shape:',labels.shape)
print('First 100 labels:',labels[:100])
nclasses = np.max(labels)+1
print('Number of classes:',nclasses)

# Show the first face for each person.
def show(i):
  plt.imshow((np.squeeze(faces[i])*255).astype('uint8'))
  plt.title('Person %d'%labels[i])
  plt.show()
for i in range(0,400,10):
  show(i)

# Resize the images to 32x32
def process(im):
  return cv.resize(np.squeeze(im),(32,32),interpolation=cv.INTER_AREA)[:,:,None]
resizedfaces = np.stack([process(im) for im in faces])

# Split into training and testing sets. We use 10% for testing.
#x_train, x_test, y_train, y_test = train_test_split(resizedfaces, labels, test_size=0.1, shuffle=True, random_state=42)
x_train = []
x_test = []
x_val = []
y_val = []
y_train = []
y_test = []
for i in range(40):
  for j in range(8):
    x_train.append(resizedfaces[i*10+j])
    y_train.append(i)
  x_val.append(resizedfaces[i*10+8])
  y_val.append(i)
  x_test.append(resizedfaces[i*10+9])
  y_test.append(i)
x_train = np.stack(x_train)
y_train = np.stack(y_train)
x_val = np.stack(x_val)
y_val = np.stack(y_val)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

# Compute the mean intensity of the training data and subtract
# it from all images. Neural networks train more effectively when the data is centered.
training_mean = np.mean(x_train)
x_train -= training_mean
x_val -= training_mean
x_test -= training_mean

# Model design
#
# Now we build the convolutional neural network model using Keras layers.
#
# We create a VGG-like network with blocks of convolution, ReLU, and max pooling.
# At the end of the network we flatten to a vector and use dense layers to reach the final 40-dimensional output.
# We use a softmax activation at the end to ensure the output probabilities sum to one.

x_in = Input(x_train.shape[1:])
# size is 32x32

# First block
x = Conv2D(8,3,padding='same')(x_in)
x = Activation('relu')(x)
x = MaxPooling2D(2,2)(x)
# size is now 16x16

# Second block
x = Conv2D(16,3,padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2,2)(x)
# size is now 8x8

# Third block
x = Conv2D(32,3,padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2,2)(x)
# size is now 4x4

# Flatten to a vector
x = Flatten()(x)

# Dense (fully-connected) layer
x = Dense(32)(x)
x = Activation('relu')(x)

# Output layer
x = Dense(nclasses)(x)
x = Activation('softmax')(x)

model = Model(inputs=x_in,outputs=x)
model.summary()

# When we compile the model we set the loss function, optimizer, and metrics to monitor during training.
#
# The "sparse categorical crossentropy" loss is appropriate when we have a classification problem and the labels are stored as integers.
# The Adam optimizer is a popular choice that works well without tuning.
# The "sparse categorical accuracy" metric will tell us the proportion of correctly classified images after each epoch.

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=3e-4),metrics=['sparse_categorical_accuracy'])

results = model.evaluate(x_train,y_train,verbose=0)
print('Training Accuracy: %.2f %%'%(results[1]*100))
results = model.evaluate(x_test,y_test,verbose=0)
print('Testing Accuracy: %.2f %%'%(results[1]*100))

# Training the model
#
# Now we are ready to train the model.
#
# To train the model, we pass the training images (X) and labels (y) to model.fit().
# shuffle=True tells Keras to randomly shuffle the data before each epoch.
# An epoch is one pass over the data. We will do 200 epochs, i.e. 200 passes over the data.
# The batch size determines how many images are shown at once to the network. A batch size of 32 is a reasonable choice.
# validation_split=0.1 tells Keras to withold 10% of the training data for validation. We only look at the metrics on this data, we don't use it to actually update the weights of the model.


history = model.fit(x_train,y_train,
          shuffle=True,
          epochs=200,
          batch_size=32,
          validation_data=(x_val,y_val))

# Here we plot the loss and accuracy curves for the training and validation data.
#
# The network reaches 100% accuracy on the training data. Naturally the network is less accurate on the
# validation data since those images are not shown to the network during training.


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss','Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Now we calculate the final accuracy on the training and test set.
results = model.evaluate(x_train,y_train,verbose=0)
print('Training Accuracy: %.2f %%'%(results[1]*100))
results = model.evaluate(x_test,y_test,verbose=0)
print('Testing Accuracy: %.2f %%'%(results[1]*100))

# model.predict() returns the output of the network. For each image we get a row of 40 probability values which add up to 1.
#
# By displaying the probabilities as an image, we should see a bright line on the diagonal if the classification was perfect.
preds = model.predict(x_test,verbose=0)
print('output shape: ',preds.shape)
print(preds[1])
plt.imshow(preds)
plt.show()

# This shows each test image with the correct and predicted label.
for im,label,pred in zip(x_test,y_test,preds):
  predlabel = np.argmax(pred)
  imout = ((np.squeeze(im)+training_mean)*255).astype('uint8')
  plt.imshow(imout)
  plt.title('Label: %d / Pred: %d'%(label,predlabel))
  plt.show()