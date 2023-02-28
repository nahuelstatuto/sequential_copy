import os
import csv
import pandas as pd
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Loss
from tensorflow import keras

from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def define_loss(d, loss_name = 'UncertaintyError'):
    if loss_name == 'UncertaintyError': 
        return UncertaintyError()
    elif d <= 2:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def params_to_vec(model):
    """docstring for params_to_vec"""
    final = []
    dims = 0
    for layer in model.layers:
        t0 = tf.reshape(layer.trainable_variables[0], [np.shape(layer.trainable_variables[0])[0]*np.shape(layer.trainable_variables[0])[1]])
        t1 = tf.reshape(layer.trainable_variables[1], [np.shape(layer.trainable_variables[1])[0]])
        final.append(tf.concat([t0,t1],0))
        dims+= np.shape(t0)[0]+np.shape(t1)[0]
    
    return final, dims

def norm_theta(vec1,vec2):    
    ''' given two parameter's vectors, return the Euclidean norm '''
    t = tf.constant(0.0, dtype = tf.float64)
    for i in range(np.shape(vec1)[0]):
        t = tf.add(t,tf.reduce_sum(tf.square(vec1[i] - vec2[i])))
    return tf.cast(t, tf.float64)
    
def evaluation(model, X, y, d, n_classes):
    ''' evaluate the model "accuracy" '''
    try:
        y_pred_ohe = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_ohe, axis=1)
    except:
        y_pred = model.predict(X)
        
    try:
        if np.shape(y)[1]==n_classes:
            y = np.argmax(y, axis=1)
    except:
        pass

    return sum(y_pred == y)/len(X)

class LambdaParameter():
    """docstring for lambda_parameter"""
    def __init__(self, lmda=0.0, automatic_lmda=False, divider=2, multiplier=1.5):
        super(LambdaParameter, self).__init__()
        self.lmda = lmda
        self.automatic = automatic_lmda
        self.divider = divider
        self.multiplier = multiplier

    def update(self, nN_prev, nN, n_sampling):
        if self.automatic:
            if (nN_prev - nN)<n_sampling:
                self.lmda = self.lmda/self.divider
            elif (nN_prev - nN)>=n_sampling:
                self.lmda = self.lmda*self.multiplier

class UncertaintyError(LossFunctionWrapper):
    '''Computes the norm between hard labels and soft predictions '''
    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="uncertainty_error"
    ):
        '''Initializes `UncertaintyError` instance.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance. Defaults to
            'uncertainty_error'.
        '''
        super().__init__(uncertainty_error, name=name, reduction=reduction)
    

def uncertainty_error(y_true, y_pred):
    ''' calculate the uncertainty error given the hard-labeled points and the predicted points '''
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)

    return tf.divide(tf.reduce_sum(tf.math.squared_difference(y_pred, y_true),1, keepdims=True), num_classes)

class MyRegularizer(regularizers.Regularizer):

    def __init__(self, layer_num, model):
        self.layer_num = layer_num
        self.model = model

    def __call__(self, x):
        self.new_weigths = self.get_weigths(self.model.layers[self.layer_num])
        self.theta0 = self.model.theta0[self.layer_num]
        return tf.divide(tf.reduce_sum(tf.square(self.theta0-self.new_weigths)),
                         tf.constant(self.model.weights_dims, dtype = tf.float64))
    
    def get_weigths(self, layer):
        #weights of the layer
        t0 = tf.reshape(layer.trainable_variables[0],
                        [np.shape(layer.trainable_variables[0])[0]*np.shape(layer.trainable_variables[0])[1]])
        #bias of the layer
        t1 = tf.reshape(layer.trainable_variables[1],
                        [np.shape(layer.trainable_variables[1])[0]])
        return tf.concat([t0,t1],0)