#!/usr/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals
from scipy import io
import numpy as np

import tensorflow as tf

def loat_data(paths):
    fs=io.loadmat(paths[0])
    arr=fs['dataset'][0][0]
    train_examples=arr
    #m is rows && n is cols
    #m*n
    m = arr.shape[0]
    n = arr.shape[1]
    train_labels=np.ones((m,1),dtype='uint8')
    i=0
    
    if len(paths)<=1:
        return train_examples,train_labels

    for path in paths[1:]:
        fs=io.loadmat(path)
        arr=fs['dataset'][0][0]
        train_examples=np.concatenate((train_examples,arr),axis=0)
        if i%2==0:
            labels=np.zeros((m,1),dtype='uint8')
        else:
            labels=np.ones((m,1),dtype='uint8')
        #append lables; ones 1000 items
        train_labels=np.concatenate((train_labels,labels),axis=0)
        i+=1
    
    return train_examples,train_labels


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64,input_dim=27301),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    
    return model


def train_test(model,all_examples,all_labels):
    #split training examples and test examples by k
    k=1400
    train_examples = all_examples[:k]
    train_labels=all_labels[:k]
    test_examples = all_examples[k:]
    test_labels = all_labels[k:]

    print(all_examples.shape)
    print(train_examples.shape)


    train_dataset=tf.data.Dataset.from_tensor_slices((train_examples,train_labels))
    test_dataset=tf.data.Dataset.from_tensor_slices((test_examples,test_labels))
    print(train_dataset)
     #shuffle and batch
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    model.fit(train_dataset,epochs=10)
    model.evaluate(test_dataset)



if __name__ == "__main__":
    paths=['featureblock_10.mat','featureblock_31.mat']

    all_examples,all_labels=loat_data(paths)

    print(type(all_examples))
    print(type(all_labels))
    print(all_examples.shape)

    '''
    #split training examples and test examples by k
    k=1400
    train_examples = all_examples[:k]
    train_labels=all_labels[:k]
    test_examles = all_examples[k:]
    test_labels = all_labels[k:]
    '''

    #train_dataset=tf.data.Dataset.from_tensor_slices((train_examples,train_labels))
    #print(train_dataset)

    model=create_model()
    train_test(model,all_examples,all_labels)

