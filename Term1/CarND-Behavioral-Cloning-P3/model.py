#/usr/bin/env python3
'''
CarND-Behavioral-Cloning-P3
'''
import cv2
import numpy as np
import tensorflow as tf
import pandas as pds
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
#from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def image_load(file_path):
    '''
    load and process image in a pipeline
    output channel (height, width, channels=3)
    '''
    # load image
    bgr_image = cv2.imread('data/'+file_path.strip()) 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def create_data_sample(csv_file, left_camera_compensation=0.1, right_camera_compensation=-0.1):
    '''
    read cvs file output train datas and vailidation datas
    '''
    # Open cvs file
    csv_data = pds.read_csv(csv_file)
    print(csv_data.shape)
    
    image_files = []
    steering_values = []
    
    # Get the 'left' 'center' 'right' 'steering' data
    data_left_camera = csv_data['left']
    data_center_camera = csv_data['center']
    data_right_camera = csv_data['right']
    data_steering = csv_data['steering']
    
    # Append to a list
    for counter in range(len(csv_data)) :
        image_files.append(data_left_camera[counter])
        steering_values.append(data_steering[counter]+left_camera_compensation)
        image_files.append(data_center_camera[counter])
        steering_values.append(data_steering[counter])
        image_files.append(data_right_camera[counter])
        steering_values.append(data_steering[counter]+right_camera_compensation) 
    
    # Shuffle the list
    image_files, steering_values = shuffle(image_files, steering_values)
    
    # Split the data to train data and validation data
    train_image_samples, validation_image_samples, train_steering_samples, validation_steering_samples = train_test_split(image_files, steering_values, test_size=0.20)
    print(len(train_image_samples), len(validation_image_samples))
    
    return train_image_samples, validation_image_samples, train_steering_samples, validation_steering_samples

    
def create_data_generator(image_samples, steering_samples, batch_size, using_flip_image='ON'):
    """
    Image generator. Returns batches of images indefinitely
    """
    current_index = 0
    features = []
    labels = []
    
    while 1 :
        
        # If loop to the end, jump to the first data, and continue the loop
        if current_index >= len(steering_samples) :
            current_index = 0
        
        # Load and preprocess the image    
        images_camera = image_load(image_samples[current_index])
        # Get the corresponding steering data
        steering_angle = steering_samples[current_index]
        
        # Append to a list
        features.append(images_camera)
        labels.append(steering_angle)
        # If using fliped image, add it to the list too
        if using_flip_image == 'ON' :
            features.append(np.fliplr(images_camera))
            labels.append(-(steering_angle))
        
        # Step one forward
        current_index += 1;
        
        # If the data size is more than batch_size, yield the data
        if(len(features) >= batch_size):
            #print(np.array(features).shape, np.array(labels).shape)
            # Shuffle the data
            features, labels = shuffle(features, labels)
            yield (np.array(features), np.array(labels))
            features = []
            labels = []

            
def create_keras_model(feature_shape):
    '''
    create mode according nivida DAVE-2
    '''
    model = Sequential()
    
    # Cropping to 70x320x3
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=feature_shape)) 
    # Normalize data
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    # Layer Convolutional. 
    model.add(Conv2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. 
    model.add(Conv2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. 
    model.add(Conv2D(48, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. 
    model.add(Conv2D(64, 3, 3, activation='relu'))
    # Layer Convolutional. 
    model.add(Conv2D(64, 3, 3, activation='relu'))
    # Layer Flatten. 
    model.add(Flatten())
    # Layer Fully Connected. 
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. 
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. 
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. Linear
    model.add(Dense(1, activation='linear'))

    return model

    
def train_model():
    '''
    Training pipeline
    '''
    csv_file_path = 'data/driving_log.csv'
    batch_size = 80 
    left_camera_compl = 0.5    # using default data
    right_camera_compl = -0.5  # using default data
    
    # Create train data and valid data
    train_features, vaild_features, train_labels, vaild_lables = create_data_sample(csv_file_path, left_camera_compensation=left_camera_compl, right_camera_compensation=right_camera_compl)
    
    # Create train and valid data generator
    train_data_generator = create_data_generator(train_features, train_labels, batch_size)
    vaild_data_generator = create_data_generator(vaild_features, vaild_lables, batch_size)
    
    # Create model
    feature_shape=(160,320,3)
    bhvcln_model = create_keras_model(feature_shape)
    bhvcln_model.summary
    #plot(bhvcln_model, to_file='model.png', show_shapes=True)
    
    bhvcln_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Save every model using Keras checkpoint
    checkpoint = ModelCheckpoint(filepath= "outputs/check-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]
                  
    # Train model
    history_object = bhvcln_model.fit_generator(generator=train_data_generator, \
                                                samples_per_epoch=38400, \
                                                nb_epoch=5, \
                                                verbose=1, \
                                                callbacks=callbacks_list, \
                                                validation_data=vaild_data_generator, \
                                                nb_val_samples=9600)
                                                
    print(history_object.history.keys())
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'vaild'], loc='upper left')
    #plt.legend(['train'], loc='upper left')
    plt.show()
    
    # Save mode
    bhvcln_model.save('model.h5')

if __name__ == '__main__':
    train_model()

