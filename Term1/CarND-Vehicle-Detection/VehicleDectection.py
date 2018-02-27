#/usr/bin/env python3
'''
Vehicle Detection image scaled and channel order
'''
import sys
import time
import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from skimage.morphology import closing, opening, erosion, dilation, square, binary_closing
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pdb

bbox_buffer = [] 

############################### Ectract Features ###############################
def get_hog_features(rgb_image, color_space='HSV', hog_channel='CH_FIRST', 
                    orient=9, pix_per_cell=8, cell_per_block=2 , 
                    vis=True, feature_vec=True):
    '''
    Function to return HOG features. Input must be RGB image, and this image will 
    be converted to target color_space, note that 'GRAY' has only one channel, so 
    using channel list to cover the problem of different channels of different color_space
    '''
    channels = []
    hog_features = []
    hog_images = []
    
    if color_space == 'GRAY':
        # Convert rgb image to gray image
        feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        channels.append(feature_image)
    else :
        feature_image = None
        # convert to target color_space
        if color_space == 'RGB':
            feature_image = np.copy(rgb_image)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
        # process the target channel
        if hog_channel == 'CH_FIRST':
            channels.append(feature_image[:,:,0])
        elif hog_channel == 'CH_SECOND':
            channels.append(feature_image[:,:,1])
        elif hog_channel == 'CH_THIRD':
            channels.append(feature_image[:,:,2])
        elif hog_channel == 'ALL':
            channels.append(feature_image[:,:,0])
            channels.append(feature_image[:,:,1])
            channels.append(feature_image[:,:,2])
    
    if vis == True:     # if we need the hog image
        # Use skimage.hog() to get both features and a visualization
        for single_channel_image in channels:
            features, hog_image = hog(single_channel_image, 
                                    orientations = orient,
                                    pixels_per_cell = (pix_per_cell, pix_per_cell),
                                    cells_per_block = (cell_per_block, cell_per_block),  
                                    transform_sqrt = False, 
                                    visualise = True, 
                                    feature_vector = feature_vec)
            hog_features = np.concatenate((hog_features, features))
            hog_images.append([hog_image, single_channel_image])  
        
        hog_features = np.ravel(hog_features)  # change it to a list         
        return hog_features, hog_images 
    else:   # if we do not need the hog image           
        # Use skimage.hog() to get features only
        for single_channel_image in channels:
            features = hog(single_channel_image, 
                        orientations = orient, 
                        pixels_per_cell = (pix_per_cell, pix_per_cell),
                        cells_per_block = (cell_per_block, cell_per_block), 
                        transform_sqrt = False, 
                        visualise = False, 
                        feature_vector = feature_vec)
            hog_features = np.concatenate((hog_features, features))
        hog_features = np.ravel(hog_features)   # change it to a list          
        return hog_features

def test_hog_feature():
    '''
    test hog feature on test images
    '''
    # Read in image
    car_features = []
    notcar_features = []
    # collect not cars data
    notcars = glob.glob('train_data/non-vehicles/Extras/*.png')
    for file_path in glob.glob('train_data/non-vehicles/GTI/*.png'):
        notcars.append(file_path)
    # collect cars data
    cars = glob.glob('train_data/vehicles/GTI_Far/*.png')
    for file_path in glob.glob('train_data/vehicles/GTI_Left/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/GTI_MiddleClose/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/GTI_Right/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/KITTI_extracted/*.png'):
        cars.append(file_path)
    print("length of cars:", len(cars))
    print("length of noncars:", len(notcars))
    
    # Generate a random index to look at a car image   
    car_ind = np.random.randint(0, len(cars))
    ind = np.random.randint(0, len(cars))
    # Read in the image
    print(cars[ind])
    car_bgr_image = cv2.imread(cars[ind]) 
    car_rgb_image = cv2.cvtColor(car_bgr_image, cv2.COLOR_BGR2RGB)  
    no_car_bgr_image = cv2.imread(notcars[ind]) 
    no_car_rgb_image = cv2.cvtColor(no_car_bgr_image, cv2.COLOR_BGR2RGB)
    car_features, car_hog_images = get_hog_features(car_rgb_image, color_space='YUV', hog_channel='ALL', vis=True, feature_vec=True)
    no_car_features, no_car_hog_images = get_hog_features(no_car_rgb_image, color_space='YUV', hog_channel='ALL', vis=True, feature_vec=True)
    #print(len(features), len(hog_images))
    for hog_image, channel_image in car_hog_images :
        # Plot result                    
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(car_rgb_image)
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(channel_image, cmap='gray')
        plt.title('Channel Image')
        plt.subplot(133)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Image')
        fig.tight_layout()
        plt.show() 
    for hog_image, channel_image in no_car_hog_images :
        # Plot result                    
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(no_car_rgb_image)
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(channel_image, cmap='gray')
        plt.title('Channel Image')
        plt.subplot(133)
        plt.imshow(hog_image, cmap='gray')
        plt.title('HOG Image')
        fig.tight_layout()
        plt.show()                

def bin_spatial(rgb_img, color_space='RGB', size=(16, 16)):
    '''
    Function to compute bin spatial features, image must in rgb format.
    '''
    # Convert image to new color space (if specified)
    if color_space == 'RGB':
        feature_image = np.copy(rgb_img)
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV) 
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)        
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features
    
def color_hist(rgb_img, color_space='RGB', nbins=32, bins_range=(0, 255)):
    '''
    Function to compute color histogram features, image must in rgb format.
    '''
    # Convert image to new color space (if specified)
    if color_space =='RGB':
        feature_image = np.copy(rgb_img)
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)   
                
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = np.ravel(hist_features) 
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    
def extract_features(rgb_image):
    '''
    Function to extract features from a list of images
    '''
    # Apply hog() to get hog features
    hog_features_1 = get_hog_features(rgb_image, color_space='YCrCb', hog_channel='CH_THIRD', vis=False, feature_vec=True)
    hog_features_2 = get_hog_features(rgb_image, color_space='YCrCb', hog_channel='CH_SECOND', vis=False, feature_vec=True)
    hog_features_3 = get_hog_features(rgb_image, color_space='YCrCb', hog_channel='CH_FIRST', vis=False, feature_vec=True)
    # Apply bin_spatial() to get spatial color features
    spatial_features_1 = bin_spatial(rgb_image, color_space='YCrCb')
    # Apply color_hist() to get color histogram features
    hist_features_1 = color_hist(rgb_image, color_space='YCrCb')
    # Return list of feature vectors
    return np.concatenate((hog_features_1, hog_features_2, hog_features_3, spatial_features_1, hist_features_1))


################################ Train Classify ################################
def train_classifier() :
    '''
    train car classifier
    '''
    car_features = []
    notcar_features = []
    # collect not cars data
    notcars = glob.glob('train_data/non-vehicles/Extras/*.png')
    for file_path in glob.glob('train_data/non-vehicles/GTI/*.png'):
        notcars.append(file_path)
    # collect cars data
    cars = glob.glob('train_data/vehicles/GTI_Far/*.png')
    for file_path in glob.glob('train_data/vehicles/GTI_Left/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/GTI_MiddleClose/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/GTI_Right/*.png'):
        cars.append(file_path)
    for file_path in glob.glob('train_data/vehicles/KITTI_extracted/*.png'):
        cars.append(file_path)
    print("length of cars:", len(cars))
    print("length of noncars:", len(notcars))
    
    # Reduce the sample size because HOG features are slow to compute
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 8000
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    
    # Iterate through the list of images
    t_start=time.time()
    for file_path in cars:
        # Read in each one by one
        bgr_image = cv2.imread(file_path.strip()) 
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Extract features
        features = extract_features(rgb_image)
        # Append the new feature vector to the features list
        car_features.append(features)
    # Iterate through the list of images
    for file_path in notcars:
        # Read in each one by one
        bgr_image = cv2.imread(file_path.strip()) 
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Extract features
        features = extract_features(rgb_image)
        # Append the new feature vector to the features list
        notcar_features.append(features)
    t_end = time.time()
    print(round(t_end-t_start, 2), 'Seconds to extract features...')  
    
    print("length of features:", len(car_features[0])) 
    print("length of car_features:", len(car_features))
    print("length of notcar_features:", len(notcar_features))
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Shuffle data 
    X, y = shuffle(X, y)
    # Split up data into randomized training and test sets by 0.2
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    print('length of train features:', len(X_train))    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train
    X_train = X_scaler.transform(X_train)
    # Apply the scaler to X_test
    X_test = X_scaler.transform(X_test)

    print('Feature vector length:', len(X_train[0]))
    print('length of train features:', len(X_train))
    
    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t_start = time.time()
    svc.fit(X_train, y_train)
    t_end = time.time()
    print(round(t_end-t_start, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    clf_pickle = {}
    clf_pickle["svc"] = svc
    clf_pickle["x_scaler"] = X_scaler

    with open('clf_pickle.p', 'wb') as f: 
        pickle.dump(clf_pickle, f)
        
    # Check the prediction time for a single sample
    t_start=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t_end = time.time()
    print(round(t_end-t_start, 5), 'Seconds to predict', n_predict,'labels with SVC')
    #return svc, X_scaler

################################## Search Car ##################################
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    '''
    Function that takes an image, start and stop positions in both x and y, window size (x and y dimensions), and overlap fraction (for both x and y)
    '''
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    '''
    if x_start_stop[0] < 0 :
        x_start_stop[0] = 0
    if x_start_stop[1] > img.shape[1]:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] < 0:
        y_start_stop[0] = 0
    if y_start_stop[1] > img.shape[0]:
        y_start_stop[1] = img.shape[0]
    '''    
    # Compute the span of the region to be searched  
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate each window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    # Return the list of windows
    return window_list

def generate_serach_windows(image) :
    '''
    generate serach windows on given image
    '''
    search_window_list = [] 
    print("image size X:", image.shape[1])
    print("image size Y:", image.shape[0])
    '''
    for counter in range(10):
        window_size = 32 * (counter+1)
        windows = slide_window(image, x_start_stop=[0, image.shape[1]], 
                    y_start_stop=[image.shape[0]/2, image.shape[0]/2+120+counter*24], 
                    xy_window=(2*window_size, window_size), xy_overlap=(0.5, 0.5))
        for window in windows :
            search_window_list.append(window)       
    print(search_window_list[0])
    print("Number of search windows:", len(search_window_list))
    '''
    xy_window_list = [(64,64), (96,96), (128,128), (256,256)]
    #x_start_stop_list = [[400, 500], [380, 500], [380, 600], [360, 700], [360, None]]
    y_start_stop_list = [[400, 560], [380, 580], [380, 650], [360, None]]
    xy_overlap_list = []
    for i in range(len(xy_window_list)):
        windows = slide_window( image, 
                                x_start_stop = [None, None], 
                                y_start_stop = y_start_stop_list[i], 
                                xy_window = xy_window_list[i], 
                                xy_overlap = (0.7, 0.7))
        for window in windows :
            search_window_list.append(window) 
    print("Number of search windows:", len(search_window_list))
    return search_window_list

def search_cars(orignal_image, search_windows, clf, scaler):
    '''
    Loop over search_windows on orignal_image, return search_windows that detect cars
    '''
    #1) Create an empty list to receive positive detection windows
    hit_windows = []
    #2) Iterate over all windows in the list
    for window in search_windows:
        #3) Extract the test window from original image
        feature_img = cv2.resize( orignal_image[
                                int(window[0][1]):int(window[1][1]), 
                                int(window[0][0]):int(window[1][0])], 
                                (64, 64), interpolation=cv2.INTER_LINEAR)      
        #4) Extract features for that window using single_img_features()
        features = extract_features(feature_img)
        #5) Scale extracted features to be fed to classifier
        window_features = scaler.transform(features.astype(np.float64).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(window_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            hit_windows.append(window)
    #8) Return windows for positive detections
    return hit_windows


############################### Augmented Display ###############################
def add_heat(orignal_image, hit_windows):
    '''
    Loop over hit_windows to create a heatmap
    '''
    heatmap = np.zeros_like(orignal_image[:,:,0]).astype(np.float)
    # Iterate through list of bboxes
    for box in hit_windows:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])] += 1
    # Return updated heatmap
    return heatmap
'''    
def apply_threshold(heatmap, threshold):
    
    #apply threshold to heatmap to get rid of the interference 
    
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap
'''

def filter_heatmap(heatmap, threshold):
    '''
    apply threshold to heatmap to get rid of the interference 
    '''
    filtered_heatmap = np.zeros_like(heatmap).astype(np.float)
    # Zero out pixels below the threshold
    filtered_heatmap[heatmap > threshold] = 255
    # Return thresholded map
    return filtered_heatmap
    
def draw_labeled_bboxes(filtered_heatmap):
    '''
    draw a box to indicate there is a car
    '''
    labels = label(filtered_heatmap)
    # Iterate through all detected cars
    bbox_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox_list.append(bbox)
    # Return the box
    return bbox_list

def augmented_display(orignal_image, hit_windows) :
    '''
    augmented display pipeline
    '''
    # Add heat to each box in box list
    heatmap = add_heat(orignal_image, hit_windows)
    # Apply threshold to help remove false positives
    filtered_heatmap = filter_heatmap(heatmap, threshold=2)
    # Find final boxes from heatmap using label function
    # Indicate the cars on image 
    open_filter_box = closing(filtered_heatmap, square(51))
    filtered_heatmap = open_filter_box
    bbox_list = draw_labeled_bboxes(filtered_heatmap)
    print('Car Number:', len(bbox_list))
    draw_img = np.copy(orignal_image)
    for bbox in bbox_list :
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 3)
    return draw_img, heatmap, filtered_heatmap
 
def augmented_display_with_filter(orignal_image, hit_windows) :
    '''
    augmented display pipeline
    '''
    global bbox_buffer
    
    filter_box = np.zeros_like(orignal_image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heatmap = add_heat(orignal_image, hit_windows)
    # Apply threshold to help remove false positives
    filtered_heatmap = filter_heatmap(heatmap, threshold=2)
    # Find boxes from heatmap using label function
    open_filter_box = closing(filtered_heatmap, square(51))
    filtered_heatmap = open_filter_box
    bbox_list = draw_labeled_bboxes(filtered_heatmap)
    print("1",bbox_list)
    # Store the last 5 bbox
    if len(bbox_buffer) < 4 :
        bbox_buffer.append(bbox_list)
    else :
        temp_buffer = bbox_buffer[1:]
        temp_buffer.append(bbox_list)
        bbox_buffer= temp_buffer
    print("2",bbox_buffer)
    # Add up last 5 box 
    for box_list in bbox_buffer :
        #print("3",box_list) 
        for box in box_list :
            #print("4",box) 
            filter_box[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])] += 1
    label_box = filter_heatmap(filter_box, 3)
    # binary_opening
    #open_filter_box = binary_closing(label_box, square(101))
    #label_box = open_filter_box
    # Find final boxes from open_filter_box using label function
    bbox_list = draw_labeled_bboxes(label_box)
    #print("5",bbox_list)
    # Indicate the cars on image
    draw_img = np.copy(orignal_image)
    print('Car Number:', len(bbox_list))
    for bbox in bbox_list :
        print("6",bbox)
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 3) 
    return draw_img, heatmap, filtered_heatmap

################################ Vehicle Detect ################################
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    '''
    Draw boxes on given image
    '''
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
   
def vehicle_detection_images(clf, scaler) :
    '''
    Detect cars on images
    '''
    # read in images
    images_path = glob.glob('test_images/*.jpg')
    print("Images number:", len(images_path))    
    # Iterate through the list of images
    for file_path in images_path:
        # Read in each one by one
        bgr_image = cv2.imread(file_path.strip()) 
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # process images
        orignal_image = rgb_image
        search_windows = generate_serach_windows(orignal_image)
        # Display search_windows
        box_image = draw_boxes(orignal_image, search_windows, color=(0, 0, 255), thick=3)
        # Windows that thought has cars 
        hit_windows = search_cars(orignal_image, search_windows, clf, scaler)
        # augmented display
        draw_img, heatmap, filtered_heatmap = augmented_display(orignal_image, hit_windows) 
        # show result
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(box_image)
        plt.title('Box Image')
        plt.subplot(133)
        plt.imshow(heatmap, cmap='gray')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
    

##################################### Run #####################################
def test_images() :
    '''
    Test on images
    '''
    clf_pickle = pickle.load( open("clf_pickle.p", "rb" ) )
    svc = clf_pickle["svc"]
    scaler = clf_pickle["x_scaler"]
    vehicle_detection_images(svc, scaler)
    
counter = 0

def test_video(segment='0') : 
    '''
    Test on images
    '''
      
    clf_pickle = pickle.load( open("clf_pickle.p", "rb" ) )
    svc = clf_pickle["svc"]
    scaler = clf_pickle["x_scaler"]
    
    def image_prcess(rgb_image):
        global counter
        # process images
        if counter < 200 * int(segment) :
            counter += 1
            return rgb_image
        #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        orignal_image = rgb_image #cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        search_windows = generate_serach_windows(orignal_image)
        hit_windows = search_cars(orignal_image, search_windows, svc, scaler)
        draw_img, heatmap, filtered_heatmap = augmented_display_with_filter(orignal_image, hit_windows) 
        return draw_img
        
    def test_video_frame(image): 
        rgb_image = image
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('test_iamge.jpg', bgr_image)
        orignal_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        print(image)
        return bgr_image
        
    #read in video
    write_output = 'output_images/project_video'+segment+'.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(image_prcess) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    
if __name__ == '__main__':
    #test_hog_feature()
    #train_classifier()
    test_images() 
    #test_video(sys.argv[1])  
    
