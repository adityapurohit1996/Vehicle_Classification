# %tensorflow_version 2.x
import tensorflow as tf
import csv 
import pdb
import numpy as np
import sklearn
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from tensorflow.keras import applications
from sklearn.model_selection import train_test_split
from random import shuffle
from math import ceil
from keras.utils.np_utils import to_categorical   
batch_size = 4

def generator(samples, batch_size=4):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        x_train = np.array([[]])
        y_train = np.array([])
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            b_first_time = True
            for batch_sample in batch_samples:
                if b_first_time:
                    # pdb.set_trace()
                    x_train = np.array([batch_sample[2],batch_sample[3],batch_sample[4],batch_sample[5],batch_sample[6]])
                    y_train = np.array(batch_sample[1])
                    b_first_time = False
                else:
                    x_train = np.vstack((x_train, [batch_sample[2],batch_sample[3],batch_sample[4],batch_sample[5],batch_sample[6]]))
                    y_train = np.vstack((y_train, batch_sample[1]))
            # y_train = to_categorical(y_train, num_classes=4)
            yield (x_train, y_train)

def train_generator(train_samples):
    return generator(train_samples)

def validation_generator(validation_samples):
    return generator(validation_samples)


model = models.Sequential()
model.add(layers.Dense(24, activation='relu', input_dim=5))
model.add(layers.Dense(24, activation='relu', input_dim=24))
model.add(layers.Dense(8, activation='relu', input_dim=24))
model.add(layers.Dense(4, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
# model.summary()

## Read CSV
csv_name = 'train_labels_12_bounds_wth_groundtruth.csv'
b_first_line = True
data = np.array([[]])
with open(csv_name) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        if b_first_line:
            data = np.array([line[0], line[1], line[2],line[3],line[4],line[5],line[6]])
            b_first_line = False
        else:
            data = np.vstack((data, [line[0], line[1], line[2],line[3],line[4],line[5],line[6]]))

train_samples, validation_samples = train_test_split(data, test_size=0.25)

history = model.fit_generator(train_generator(train_samples),
                            steps_per_epoch=ceil(len(train_samples)/batch_size),
                            validation_data=validation_generator(validation_samples),
                            validation_steps=ceil(len(validation_samples)/batch_size),
                            epochs=10, 
                            verbose=1)
model.save('simple_nn.h5')
