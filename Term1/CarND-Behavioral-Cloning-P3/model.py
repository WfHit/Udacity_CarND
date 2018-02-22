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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def _normalize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
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
    70 rows pixels from the top of the image
    20 rows pixels from the bottom of the image
    0 columns of pixels from the left of the image
    0 columns of pixels from the right of the image
    channel type = (height, width, channels)
    """
    img_roi_x      = 70  
    img_roi_y      = 0  
    img_roi_height = image_data.shape[0] - 70 - 20
    img_roi_width  = image_data.shape[1]                                                                       
    return image_data[img_roi_x:(img_roi_x+img_roi_height),img_roi_y:(img_roi_y+img_roi_width)]


def image_preprocess(file_path):
    '''
    load and process image in a pipeline
    output channel (height, width, channels=3)
    '''
    # 1. load image
    rgb_image = mpimg.imread('data/'+file_path.strip())
    # 2. turn image to gray
    #gray_image = cv2.cvtColor(origan_image, cv2.COLOR_BGR2GRAY)
    # 3. cropping image
    #cropped_image = _cropping_image(rgb_image)
    # 4. resizing image 
    #resized_image = cv2.resize(cropped_image, (320, 108), interpolation=cv2.INTER_LINEAR)
    # 5. normalize image
    #normalize_image = _normalize_image(resized_image)
    # 6. reshape
    #image_size = normalize_image.shape
    #reshape_image = np.reshape(normalize_image, (image_size[0], image_size[1], 3) )
    #print(reshape_image.shape)
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
        images_camera = image_preprocess(image_samples[current_index])
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
    create mode according nivida DAVE2
    '''
    model = Sequential()
    
    # Cropping to 70x320x3
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=feature_shape)) 
    # Normalize data
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    # Layer Convolutional. Input = 320x108x3. Output = 158x52x24.
    model.add(Conv2D(24, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. Input = 158x52x24. Output = 77x24x36.
    model.add(Conv2D(36, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. Input = 77x24x36. Output = 36x10x48.
    #model.add(Conv2D(48, 5, 5, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. Input = 36x10x48. Output = 17x4x64.
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Convolutional. Input = 17x4x64. Output = 7x1x64
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Layer Flatten. Input = 7x1x64. Output = 448.
    model.add(Flatten())
    # Layer Fully Connected. Input = 448. Output = 1164.
    model.add(Dense(1164, activation='relu'))#1164
    model.add(Dropout(0.5))
    # Layer Fully Connected. Input = 1164. Output = 100.
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. Input = 100. Output = 50.
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. Input = 50. Output = 10.
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    # Layer Fully Connected. Linear
    model.add(Dense(1, activation='linear'))

    return model

    
def train_model():
    '''
    CarND-Behavioral-Cloning-P3
    '''
    csv_file_path = 'data/driving_log.csv'
    batch_size = 80 
    #left_camera_compl = 0.2    # using default data
    #right_camera_compl = -0.2  # using default data
    
    # Create train data and valid data
    train_features, vaild_features, train_labels, vaild_lables = create_data_sample(csv_file_path)
    
    # Create train and valid data generator
    train_data_generator = create_data_generator(train_features, train_labels, batch_size)
    vaild_data_generator = create_data_generator(vaild_features, vaild_lables, batch_size)
    
    # Create keras model
    feature_shape=(160,320,3)
    bhvcln_model = create_keras_model(feature_shape)
    bhvcln_model.summary
    
    bhvcln_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Save every model using Keras checkpoint
    checkpoint = ModelCheckpoint(filepath= "outputs/check-{epoch:02d}-{val_loss:.4f}.hdf5", verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]
                  
    # Train model
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
    
    # Save mode
    bhvcln_model.save('model.h5')

if __name__ == '__main__':
    train_model()

