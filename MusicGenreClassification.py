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
                
                

    
            
            
            
            
            
            
            
