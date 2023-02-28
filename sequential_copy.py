import os
import logging
import numpy as np
import tensorflow as tf
from keras.utils import losses_utils

from utils import LambdaParameter
from models import params_to_vec
import plots as pt

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sequential_train(model, 
                     sampler, 
                     original, 
                     n_samples_iter=500, 
                     max_iter=1, 
                     epochs=1000, 
                     batch_size=32, 
                     max_subtrain=2,
                     n_classes=2,
                     sample_selection=True,
                     thres=1e-9,
                     lmda_par = LambdaParameter(), 
                     verbose=False,
                     X_test=None,
                     y_test=None,
                     plot_every=False):
    
    t = 0  
    lr = model.optimizer.lr.numpy()
    rho_max = tf.constant(1.0, dtype = tf.float64)
    n_subtrain = 0
    
    X_train, y_train = np.empty((0, sampler.d)), np.empty((0), dtype=int)
    
    while t < max_iter: 
        
        X, y = sampler.get_samples(original, n_samples_iter)
        X_train, y_train = np.vstack((X, X_train)), np.append(y, y_train)
        nN_prev=len(X_train)
        
        if sample_selection:
            X_train, y_train = sample_selection_policy(model, X_train, y_train, sampler.d, n_classes, thres)            
        
        model.n.append(len(X_train))
        
        lmda_par.update(nN_prev, len(X_train), n_samples_iter)
        lmda = lmda_par.lmda
        
        y_errors = y_train
        rho_mean = model.loss(tf.one_hot(y_train, n_classes), model.predict(X_train, verbose=0)).numpy()
        
        model.theta0, model.weights_dims = params_to_vec(model, return_dims=True)
        
        while len(y_errors)!=0 and n_subtrain<=max_subtrain:
            '''print('t:', t, '-- subtrain:', n_subtrain)
            print('lerning rate (base): ', lr)
            print('lerning rate - pre: ', model.optimizer.lr)
            '''
            if n_subtrain > 0:
                model.optimizer.lr=model.optimizer.lr*(n_subtrain-1+1)/(n_subtrain-1+0.5)
            #print('lerning rate - post: ', model.optimizer.lr)
            
            y_ohe = tf.one_hot(y_train, n_classes) 
            model.fit(X_train, 
                      y_ohe, 
                      lmda, 
                      rho_max, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      verbose=0)
            
            rho_max = model.loss(tf.one_hot(y_train, n_classes), model.predict(X_train, verbose=0)).numpy()
            
            y_pred_ohe = model.predict(X_train, verbose=0)
            y_pred = np.argmax(y_pred_ohe, axis=1)
            X_errors = X_train[y_pred!=y_train,:]
            y_errors = y_train[y_pred!=y_train]
            
            '''print('N errors:',len(X_errors))
            print('acc_test',evaluation(model, X_test, y_test, sampler.d, n_classes))
            print('acc_train',evaluation(model, X_train, y_train, sampler.d, n_classes))
            '''
            
            n_subtrain +=1
        
        '''print('lerning rate (base): ', lr)
        print('lerning rate del modelo: ', model.optimizer.lr)
        '''
        model.optimizer.lr = lr
        n_subtrain = 0
        t += 1
        
        if X_test is not None:
            acc_test = evaluation(model, X_test, y_test, sampler.d, n_classes)
            model.acc_test.append(acc_test)
        acc_train = evaluation(model, X_train, y_train, sampler.d, n_classes)
        model.acc_train.append(acc_train)
        model.rho.append(rho_max)
        model.lmda_vector.append(lmda)
        if plot_every:
            pt.plot_copy_model(model,X_train,y_train,X_errors,y_errors)
        
    return model

def evaluation(model, X, y,d,n_classes):
    ## evaluate the model "accuracy"
    try:
        y_pred_ohe = model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_ohe, axis=1)
    except:
        y_pred = model.predict(X, verbose=0)
        
    try:
        if np.shape(y)[1]==n_classes:
            y = np.argmax(y, axis=1)
    except:
        pass

    return sum(y_pred == y)/len(X)

def sample_selection_policy(model, X_train, y_train, d, n_classes, thresh):
        X = np.empty((0, d))
        y = np.empty((0), dtype=int)
        y_pred = model.predict(X_train, verbose=0)
        model.loss.reduction = losses_utils.ReductionV2.NONE
        rho = model.loss(tf.one_hot(y_train, n_classes), y_pred).numpy()
        model.loss.reduction = losses_utils.ReductionV2.AUTO

        for i, r in enumerate(rho>=thresh):
            if r:
                y = np.append(y,y_train[i])
                X = np.append(X,[X_train[i]], axis=0)
        if len(X)==0:
            try:
                nN = np.random.randint(0,len(X_train),int(len(X_train)/2))
                for n in nN:
                    y = np.append(y,y_train[n])
                    X = np.append(X,[X_train[n]], axis=0)
            except:
                pass
        return X, y
