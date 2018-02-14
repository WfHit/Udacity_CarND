#/usr/bin/env python3
'''
CarND-Behavioral-Cloning-P3
'''
import cv2
import numpy as np
import tensorflow as tf
import pandas as pds
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def _normalize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    image_min = 0
    image_max = 255
    image_data.astype(float)
    return a + ( ( (image_data - image_min)*(b - a) )/( image_max - image_min ) )

def _cropping_image(image_data):
    """
    50 rows pixels from the top of the image
    20 rows pixels from the bottom of the image
    0 columns of pixels from the left of the image
    0 columns of pixels from the right of the image
    channel type = (height, width, channels)
    """
    img_roi_x      = 50  
    img_roi_y      = 0  
    img_roi_height = image_data.shape[0] - 50 - 20
    img_roi_width  = image_data.shape[1]                                                                       
    return image_data[img_roi_x:(img_roi_x+img_roi_height),img_roi_y:(img_roi_y+img_roi_width)]


def image_preprocess(file_path):
    '''
    load and process image
    output channel (height, width, channels=3)
    '''
    # 1. load image
    origan_image = cv2.imread(file_path.strip())
    # 2. turn image to gray
    #gray_image = cv2.cvtColor(origan_image, cv2.COLOR_BGR2GRAY)
    # 3. cropping image
    #cropped_image = _cropping_image(origan_image)
    # 5. normalize image
    normalize_image = _normalize_image(origan_image)
    # 6. reshape
    image_size = normalize_image.shape
    reshape_image = np.reshape(normalize_image, (image_size[0], image_size[1], 3) )
    #print(reshape_image.shape)
    return reshape_image


def create_data_sample(csv_file, left_camera_compensation=0.2, right_camera_compensation=-0.2):
    '''
    read cvs file output train datas and vailidation datas
    '''
    csv_data = pds.read_csv(csv_file)
    print(csv_data.shape)
    
    image_files = []
    steering_values = []
    
    data_left_camera = csv_data['left']
    data_center_camera = csv_data['center']
    data_right_camera = csv_data['right']
    data_steering = csv_data['steering']
    
    for counter in range(len(csv_data)) :
        image_files.append(data_left_camera[counter])
        steering_values.append(data_steering[counter]+left_camera_compensation)
        image_files.append(data_center_camera[counter])
        steering_values.append(data_steering[counter])
        image_files.append(data_right_camera[counter])
        steering_values.append(data_steering[counter]+right_camera_compensation)
    
    image_files, steering_values = shuffle(image_files, steering_values)
    
    train_image_samples, validation_image_samples, train_steering_samples, validation_steering_samples = train_test_split(image_files, steering_values, test_size=0.20)
    print(len(train_image_samples), len(validation_image_samples))
    
    return train_image_samples, validation_image_samples, train_steering_samples, validation_steering_samples

    
def create_data_generator(image_samples, steering_samples, batch_size):
    """
    Image generator. Returns batches of images indefinitely
    - path : path to csv file
    - batch_size : batch size
    """
    current_index = 0
    features = []
    labels = []
    
    while 1 :
        
        if current_index >= len(steering_samples) :
            current_index = 0
            
        images_camera = image_preprocess(image_samples[current_index])
        steering_angle = steering_samples[current_index]
        
        features.append(images_camera)
        labels.append(steering_angle)
        #features.append(np.fliplr(images_camera))
        #labels.append(-(steering_angle))
        
        current_index += 1;
        
        if(len(features) >= batch_size):
            #print(np.array(features).shape, np.array(labels).shape)
            yield (np.array(features), np.array(labels))
            features = []
            labels = []

            
def create_keras_model(feature_shape):
    '''
    create mode according nivida DAVE2, but drop the fifth layer due to the picture size
    '''
    model = Sequential()
    
    #
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=feature_shape)) 
    #
    model.add(Conv2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    #
    model.add(Conv2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    #
    model.add(Conv2D(48, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    #
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    #
    #model.add(Conv2D(64, 3, 3, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    #
    model.add(Flatten())
    #
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    #
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    #
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    #
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    #
    model.add(Dense(1, activation='linear'))

    return model

    
def train_model():
    '''
    CarND-Behavioral-Cloning-P3
    '''
    csv_file_path = 'driving_log.csv'
    batch_size = 80 
    #left_camera_compl = 0.2
    #right_camera_compl = -0.2
    
    train_features, vaild_features, train_labels, vaild_lables = create_data_sample(csv_file_path)
    
    train_data_generator = create_data_generator(train_features, train_labels, batch_size)
    vaild_data_generator = create_data_generator(vaild_features, vaild_lables, batch_size)
    
    feature_shape=(160,320,3)
    bhvcln_model = create_keras_model(feature_shape)
    bhvcln_model.summary
    
    bhvcln_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    #save every model using Keras checkpoint
    checkpoint = ModelCheckpoint(filepath= "check-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]
                  
    # train model
    history_object = bhvcln_model.fit_generator(generator=train_data_generator, \
                                                samples_per_epoch=38400, \
                                                nb_epoch=10, \
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
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    bhvcln_model.save('model.h5')

if __name__ == '__main__':
    train_model()

