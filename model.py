import cv2
import numpy as np
import csv
import os
import sklearn
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.activations import relu
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


def get_samples(dirname):
    '''
    The function that extracts samples from the folders
    '''
    samples = []
    filename1 = dirname + '/driving_log.csv'
    with open(filename1) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line.append(dirname)
            samples.append(line)
    samples = samples[1:]
    return samples
	

def generator(samples, batch_size = 100):
    '''
    The generator function that yeilds data; this function is called
    by model.fit_generator()
    '''
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, angles = [], []
            for batch_sample in batch_samples:
                name = batch_sample[-1] + '/IMG/' + batch_sample[0].split('/')[-1]
                name_left = batch_sample[-1] + '/IMG/' + batch_sample[1].split('/')[-1]
                name_right = batch_sample[-1] + '/IMG/' + batch_sample[2].split('/')[-1]
                names = [name, name_left, name_right]
                center_angle = float(batch_sample[3])
                res = img_add(names, center_angle, 0.2, False)
                res_flip = img_add(names, center_angle, 0.2, True)
                images = images + res[0]
                angles = angles + res[1]
                images = images + res_flip[0]
                angles = angles + res_flip[1]
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def img_add(names, angle, correction, flip):
    '''
    The function that creates more data and flipping it
    '''
    images = []
    angles = []
    for i,name in enumerate(names):
        image = cv2.imread(name)
        # cropping top 60 and bottom 25 pixels
        image = image[60:-25, :, :]
        # resize to 200x66
        image = cv2.resize(image, (200, 66), cv2.INTER_AREA)
        # convert to YUV (as mentioned in Nvidia paper)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        if flip:
            image = np.fliplr(image)
        if i == 0:
            if flip:
                anglec = -angle
            else:
                anglec = angle
            images.append(image)
            angles.append(anglec)
        if i == 1:
            if flip:
                angle1 = -angle - correction
            else:
                angle1 = angle + correction
            images.append(image)
            angles.append(angle1)
        if i == 2:
            if flip:
                angle2 = -angle + correction
            else:
                angle2 = angle - correction
            images.append(image)
            angles.append(angle2)
    return images, angles
	

def nvidia_model():
    '''
    Nvidia model from the paper with dropout
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape= (66,200,3)))
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), border_mode ='valid', activation = 'relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), border_mode ='valid', activation = 'relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), border_mode ='valid', activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, subsample = (1,1), border_mode ='valid', activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, subsample = (1,1), border_mode ='valid', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model
    

# Dataset used
#dirname_udacity = 'udacity/data'
#dirname_recover = 'recover/data'
#dirname_recover2 = 'recover2/data'
samples1 = get_samples(dirname_udacity)
samples2 = get_samples(dirname_recover)
samples3 = get_samples(dirname_recover2)
# Initial set of images
samples = samples1 + samples2 + samples3
# split data into training/validation 80/20
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
# the generator function that generates training data
train_generator = generator(train_samples, batch_size = 100)
# the generator function that generates validation data
validation_generator = generator(validation_samples, batch_size = 100)
# model
model = nvidia_model()
# optimization using Adam
model.compile(loss = 'mse', optimizer = 'adam')
checkpoint = ModelCheckpoint('output/model1-{epoch:03d}.h5',monitor='val_loss',verbose=0,save_best_only=False, mode='auto')
history_object = model.fit_generator(train_generator, samples_per_epoch = 600, validation_data = validation_generator, 
nb_val_samples = len(validation_samples), nb_epoch = 5, callbacks=[checkpoint])
# save trained model as model.h5
model.save('output/model1.h5')