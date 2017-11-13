# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:40:18 2017

@author: laljarus
"""

# Import Test Data

# Load pickled data
import pickle
import numpy as np
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data

training_file = './TestData/train.p'
validation_file= './TestData/valid.p'
testing_file = './TestData/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[1].shape 

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train)+1

assert max(y_train)==max(y_valid), 'The number of classes are different between training and validation set'
assert max(y_train)==max(y_test), 'The number of classes are different between training and Test set'

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.


sign_names_file = "./signnames.csv"

f = open('./signnames.csv','r')
lines = f.readlines()
dictSignnames = {}
for line in lines:
    text = line.split(',')
    dictSignnames.update({text[0]:text[1]})   

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(dictSignnames.get(str(y_train[index])))

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

X_train = (X_train -128)/128
X_valid = (X_valid - 128)/128
X_test = (X_test -128)/128

print('Data is normalized')

### Define your architecture here.
### Feel free to use as many code cells as needed.
# Shuffling the data
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 15
BATCH_SIZE = 128

#LeNet function

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1        
        
    weights = {'wc1':tf.Variable(tf.truncated_normal([5,5,3,6],mu,sigma,dtype=tf.float32)),\
               'wc2':tf.Variable(tf.truncated_normal([5,5,6,16],mu,sigma,dtype=tf.float32)),\
               'wd1':tf.Variable(tf.truncated_normal([400,120],mu,sigma,dtype=tf.float32)),\
               'wd2':tf.Variable(tf.truncated_normal([120,84],mu,sigma,dtype=tf.float32)),\
               'wd3':tf.Variable(tf.truncated_normal([84,n_classes],mu,sigma,dtype=tf.float32))\
              }
    
    biases = {'bc1': tf.Variable(tf.zeros([6],dtype=tf.float32)),\
              'bc2': tf.Variable(tf.zeros([16])),\
              'bd1': tf.Variable(tf.zeros([120])),\
              'bd2': tf.Variable(tf.zeros([84])),\
              'bd3': tf.Variable(tf.zeros([n_classes])),\
             }
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = tf.nn.conv2d(x,weights['wc1'],strides=[1,1,1,1],padding='VALID')
    conv1 = tf.nn.bias_add(conv1,biases['bc1'])    

    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    
    #conv1_dims = [conv1.shape.dims[1].value,conv1.shape.dims[2].value,conv1.shape.dims[3].value]
    #assert conv1_dims == [28,28,6],'The ouput of first convolution layer is not right'

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #pool1_dims = [pool1.dims[1].value,pool1.dims[2].value,pool1.dims[3].value]
    #assert pool1_dims == [14,14,6],'The ouput of first pooling layer is not right'

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2 = tf.nn.conv2d(pool1,weights['wc2'],strides=[1,1,1,1],padding='VALID')
    conv2 = tf.nn.bias_add(conv2,biases['bc2'])    
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    #conv2_dims = [conv2.shape.dims[1].value,conv2.shape.dims[2].value,conv2.shape.dims[3].value]
    #assert conv1_dims == [10,10,16],'The ouput of first convolution layer is not right'

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'VALID')
    
    #pool1_dims = [pool2.dims[1].value,pool2.dims[2].value,pool2.dims[3].value]
    #assert pool1_dims == [5,5,16],'The ouput of first pooling layer is not right'

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat1 = flatten(pool2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(flat1,weights['wd1']),biases['bd1'])
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1,weights['wd2']),biases['bd2'])
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2,weights['wd3']),biases['bd3'])
    
    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
# Normalizing the data with zero mean 

# Placeholder for inputs and labels

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# Forward and Backprobagation

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#Evaluating the Model

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Training the model 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './TrafficSignClassifier')
    print("Model saved")

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
            

