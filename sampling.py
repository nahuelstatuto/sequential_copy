import os
import numpy as np
import tensorflow as tf
import logging

from numpy import genfromtxt
from numpy import savetxt

class Sampler():
    """docstring for ... """ 
    
    data = None
    iteration = 0
    file = None
       
    def __init__(self, distr='Gaussian', d=2, n_classes=2, std=1.5, from_file=False,
                 to_file=False, file_path=None, automatic_fill=False, balancer=False, target=-1):
        self.d = d
        self.n_classes = n_classes
        self.std = std
        self.distr = distr
        self.target = target
        
        self.from_file = from_file
        self.to_file = to_file
        self.file_path = file_path
        self.automatic_fill = automatic_fill
        self.balancer = balancer
        
        self.set_file()
        
        
    def set_file(self):
        if self.from_file:
            if self.file_path:
                self.file = self.open_file_for_read()
            else:
                logging.error('No such file: \'{}\'. Define \'file_path\'.'.format(self.file_path))
                self.from_file, self.to_file, self.automatic_fill  = False, False, False
        
    def open_file_for_read(self):
        try:
            return open(os.path.join(self.file_path), 'rb')
        except:
            self.from_file = False
            if self.automatic_fill:
                fp = self.create_sampling_file()
                return fp
            else:
                logging.error('No such file: \'{}\'. Set \'automatic_fill\' = True to create it.'.format(self.file_path))
                self.to_file = False
                return None

    def create_sampling_file(self):
        try:
            fp = open(os.path.join(self.file_path), 'wb')
            self.to_file = True
            logging.warning('No such file: \'{}\'. File created!'.format(self.file_path))
            return fp
        except:
            logging.error('No file_path defined. File can not be created.')
            return None
            
    def close_file(self):
        if self.file:
            self.file.close()        
        
    def get_samples(self, original, num_samples = 100):
        if self.from_file and self.file:
            X, y = self.read_samples_from_file(original, num_samples)
        elif self.from_file and not(self.file):
            logging.error('Define \'file_path\'.'.format(self.file_path))
            self.from_file, self.to_file, self.automatic_fill = False, False, False
            return np.empty((0, self.d)), np.empty((0), dtype=int)
        else:
            X, y = self.generate_samples(original, num_samples)
            if self.to_file:
                self.add_samples_to_file(X, y)
            
        return X, y

    def read_samples_from_file(self, original, num_samples):
        
        if self.iteration==0:
            self.data = genfromtxt(self.file, delimiter=',') 
        X_new = np.empty((0, self.d))
        y_new = np.empty((0), dtype=int)
            
        for line in self.data[self.iteration*num_samples:(self.iteration+1)*num_samples]:
            X_new, y_new = np.vstack((X_new,np.asarray(line[:self.d]))), np.append(y_new,line[self.target])

        self.iteration+=1
        
        if len(X_new)==num_samples:
            return X_new, y_new 

        logging.warning('Not enough data points from file. {} data points were generated.'.format(num_samples-len(X_new)))
        self.from_file, self.to_file = False, True  
        
        X, y = self.generate_samples(original, num_samples-len(X_new))
        self.add_samples_to_file(X, y)      
        X_new, y_new = np.vstack((X_new,X)), np.append(y_new,y)
        
        return X_new, y_new

    def generate_samples(self, original, num_samples):
        if self.balancer:
            X_new, y_new = rBalancer(N=0, d=self.d, K=self.n_classes, model=original,
                                     max_iter=10, N_batch=num_samples, low=0,
                                     high=2*np.sqrt(self.d))
        elif self.distr == 'Gaussian':
            X_new = np.random.multivariate_normal(np.zeros((self.d,)),
                                                  np.eye(self.d,self.d)*self.std,
                                                  size=num_samples)
        else:
            logging.warning('Distribution \'{}\' does not exist. Automatically changed to Gaussian!'.format(self.distr))
            self.distr = 'Gaussian'
            return self.generate_samples(original, num_samples)
        
        y_new = original.predict(X_new)
        return X_new, y_new
    
    def add_samples_to_file(self, X, y):
        self.file = open(os.path.join(self.file_path), 'a+')
        for X_,y_ in zip(X,y):            
            savetxt(self.file, np.asarray([ np.append(X_[:],y_)]), delimiter=',')
        self.close_file()


def rBalancer(N,d,K,model,max_iter=10,N_batch=100000, low=0, high=1):
    # N is the amount of samples per class required
    # max_iter is the max number of iterations to get to N
    # K is the number of classes in the problem
    # N_batch is the number of elements sampled at each iteration
    bins = np.arange(K+1)-0.5
    classes = np.arange(K)
    #Generate random direction
    v = np.random.multivariate_normal(np.zeros((d,)),np.eye(d,d),size = N_batch)
    v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
    #Scale the direction between low and high
    #print('Generating between => low: '+str(low)+' and high: '+str(high))
    alpha = np.random.uniform(low=low,high=high,size = N_batch)
    #alpha = np.random.exponential(scale=5.,size=N_batch)

    qsynth = np.dot(alpha[:,np.newaxis],np.ones((1,d)))*v
    y_synth = model.predict(qsynth)
    nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
    #print(np.unique(y_synth))
    #print(nSamplesPerClass)
    toAdd = classes[nSamplesPerClass<N]
    #print(toAdd)
    #print(nSamplesPerClass[toAdd])
    for i in range(max_iter):
        #Generate random direction
        v = np.random.multivariate_normal(np.zeros((d,)),np.eye(d,d),size = N_batch)
        v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
        #print('Generating between => low: '+str(low)+' and high: '+str(high))
        #alpha = np.random.exponential(scale=0.1,size=N_batch)
        alpha = np.random.uniform(low=low,high=high,size = N_batch)
        qtmp = np.dot(alpha[:,np.newaxis],np.ones((1,d)))*v
        y_synth = model.predict(qtmp)

        #Select samples to add
        idx = [i for i in range(N_batch) if y_synth[i] in toAdd]
        #Add samples to the synthetic set
        qsynth = np.r_[qsynth,qtmp[idx,:]]
        y_synth = model.predict(qsynth)

        nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
        toAdd = classes[nSamplesPerClass<N]
        #print('To ADD:' + str(i)+str(toAdd))
        #print(nSamplesPerClass[toAdd])
        if len(toAdd)<1:
            return qsynth,y_synth
    return qsynth,y_synth
