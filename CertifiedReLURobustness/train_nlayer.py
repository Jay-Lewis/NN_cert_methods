## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os
from save_nlayer_weights import NLayerModel

def train(inputs, labels, num_classes=10, model=None,params=None, num_epochs=50, batch_size=128, train_temp=1, init=None, lr=0.01, decay=1e-5, momentum=0.9):
    """
    Train a n-layer simple network for MNIST and CIFAR
    """
    
    # # create a Keras sequential model
    if model is None:
        nlayer_model = NLayerModel(params, num_classes=num_classes)
        model = nlayer_model.model

    # load initial weights when given
    if init != None:
        model.load_weights(init)

    # define the loss function which is the cross entropy between prediction and true label
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    # initiate the SGD optimizer with given hyper parameters
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    # compile the Keras model, given the specified loss and optimizer
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.summary()
    print('=================')
    print("training")
    print('=================')
    # run training with given dataset, and print progress
    history = model.fit(inputs, labels,
              batch_size=batch_size,
              validation_data=(inputs, labels),
              epochs=num_epochs,
              shuffle=True)
    

    # # save model to a file
    # if file_name != None:
    #     model.save(file_name)
    print('=================')
    print('finished training')
    print('==================')
    return {'model': nlayer_model, 'history': None}

if not os.path.isdir('models'):
    os.makedirs('models')


if __name__ == '__main__':
    print(MNIST().train_data.shape[1:])
    train(MNIST(), file_name="models/mnist_5layer_relu", params=[20,20,20,20], num_epochs=50, lr=0.02, decay=1e-4)

