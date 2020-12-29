# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 11:09:46 2020

@author: Mousam
"""


import numpy as np
import math
import librosa
import os
import json
from sklearn.model_selection import train_test_split

DATASET_PATH = 'genres'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
SAMPLE_PER_TRACK = SAMPLE_RATE * 30

# hop_length = number of samples between successive frames
def mfcc(dataset_path, json_path, n_mfcc = 15, n_fft = 2048, hop_length = 512, num_segments = 10):
    data = {
        'genre' : [],
        'mfcc' : [],
        'labels' : []
        }
    
    num_samples_per_segment = SAMPLE_PER_TRACK // num_segments
    # math.ceil() func rounds up the value to next large integer
    mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    # OS.walk() generate the file names in a directory tree by walking the tree 
    # either top-down or bottom-up
    for i, (dirpath, dirname, filename) in enumerate(os.walk(dataset_path)):
        
        # in some iteration over dataset_path, os.walk() returns dataset_path as dirpath
        # so we should avoid the root 
        if dirpath is not dataset_path:
            
            # save the genres in the dictionary
            components = dirpath.split(',')
            semantic_label = components[-1]
            data['genre'].append(semantic_label)
            
            # extracting audio files
            for f in filename:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, SAMPLE_RATE)
                
                # process segment extracting mfcc 
                for x in range(num_segments):
                    start_sample = num_samples_per_segment * x
                    end_sample = start_sample + num_samples_per_segment
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample:end_sample],
                                                sr = SAMPLE_RATE,
                                                n_mfcc = n_mfcc)
                    
                    mfcc = mfcc.T
                    
                    if len(mfcc) == mfcc_vector_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        print('filepath:{}, segment: {}'.format(file_path, x))
                        
                        
                        
                        
    with open(json_path, 'w') as fp:
        json.dump(data, fp)
        
def loadData(DATASET_PATH):
    with open(DATASET_PATH, 'r') as fp:
        dataset = json.load(fp)
        
        mfcc_arr = np.array(dataset['mfcc'])
        label_arr = np.array(dataset['labels'])
        
    return mfcc_arr, label_arr
                
                
if __name__ == '__main__' :
    num_segments = 10
    mfcc(DATASET_PATH, JSON_PATH, num_segments)  

    # load the data              
    inputs, targets = loadData('data.json')
    
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size = 0.3, random_state = 0)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    import tensorflow as tf
    # BUILD A NEURAL NETWORK 
    model = Sequential()
    model.add(Flatten(input_shape = (inputs.shape[1], inputs.shape[2])))
        
    model.add(Dense(units = 512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dense(units = 128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(units = 512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    # adding output layer
    model.add(Dense(units = 10, activation='softmax'))
        
          
    # compile the network
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    # run the model
    model.fit(X_train, y_train, batch_size=32, epochs = 70, validation_data=(X_test, y_test))
    model.summary()
    
    
            
            
            
            
            
            
            