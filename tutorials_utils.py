import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

# imports from our libreries
from sampling import Sampler
from utils import define_loss
from utils import LambdaParameter
from models import FeedForwardModel
from sequential_copy import sequential_train

def separate_runs(original,
                  lr=0.0005,
                  n_samples_iter=100, 
                  thres=1e-9, 
                  lmda_par = 0.0,
                  max_iter=3, 
                  X_test=None, 
                  y_test=None, 
                  layers = [64,32,10],
                  sample_selection=True):
    
    d = X_test.shape[1]
    n_classes = len(np.unique(y_test))
    
    # define optimizer and loss 
    opt_ = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_ = define_loss(d, loss_name = 'UncertaintyError')

    # define new model
    seq_copy_ = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
    seq_copy_.build(input_shape=(layers[0],d))
    seq_copy_.compile(loss=loss_, optimizer=opt_)

    # define the memory (lambda) parameter
    lmda_ = LambdaParameter(lmda=lmda_par, automatic_lmda=False)

    #define the sampling process
    sampler_ = sp.Sampler(d=d, n_classes=n_classes)
    
    return sequential_train(seq_copy_, 
                            sampler_, 
                            original, 
                            n_samples_iter=n_samples_iter, 
                            thres=thres, 
                            lmda_par =lmda_,
                            max_iter=max_iter, 
                            X_test=X_test, 
                            y_test=y_test, 
                            sample_selection=sample_selection,
                            plot_every=False)

def decode_results(data):
    n, acc_train, acc_test, rho, lmda = ([] for i in range(5))
    
    for i in range(len(data)):
        n.append(list(data[i].n))
        acc_train.append(list(data[i].acc_train))
        acc_test.append(list(data[i].acc_test))
        rho.append(list(data[i].rho))
        lmda.append(list(data[i].lmda_vector))
        
    return np.array(n), np.array(acc_train), np.array(acc_test), np.array(rho), np.array(lmda)


def plot_results(model, plot_type='acc_test'):
    fig = plt.figure(figsize=(6,3)) 
    ax = fig.add_subplot(111)    
    ax.set_xlabel('Iteration')
    x = np.arange(1, len(trained_model.acc_test)+1)
    ax.set_xticks(np.arange(1, len(trained_model.acc_test)+1))
    
    if plot_type == 'acc_test':
        plt.plot(x, trained_model.acc_test,'o--')
        ax.set_ylabel('Acc test')
    elif plot_type == 'nN':
        plt.plot(x, trained_model.n,'o--')
        ax.set_ylabel('Number of synthetic data points')
    elif plot_type == 'rho':
        plt.plot(x, trained_model.rho,'o--')
        ax.set_ylabel('Average uncertainty rho')
    else:
        pass
    
    plt.show()