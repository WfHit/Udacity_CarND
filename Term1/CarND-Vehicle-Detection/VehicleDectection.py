#/usr/bin/env python3
'''
'''
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
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pdb

############################### Ectract Features ###############################

# Define a function to return HOG features
# Input must be RGB image
def get_hog_features(rgb_image, color_space='GRAY', hog_channel='ALL', orient=9, 
                    pix_per_cell=8, cell_per_block=2 , 
                    vis=True, feature_vec=True):
    channels = []
    hog_features = []
    hog_images = []
    #pdb.set_trace()
    if color_space == 'GRAY':
        # Convert rgb image to gray image
        feature_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        channels.append(feature_image)
    else :
        feature_image = None
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
    
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        for single_channel_image in channels:
            features, hog_image = hog(single_channel_image, orientations=orient,
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block), 
                                    transform_sqrt=False, visualise=True, 
                                    feature_vector=feature_vec)
            hog_features = np.concatenate((hog_features, features))
            hog_images.append(hog_image)
            
        hog_features = np.ravel(hog_features)   
        
        return hog_features, hog_images

    else:      
        # Use skimage.hog() to get features only
        for single_channel_image in channels:
            features = hog(single_channel_image, orientations=orient, 
                        pixels_per_cell=(pix_per_cell, pix_per_cell),
                        cells_per_block=(cell_per_block, cell_per_block), 
                        transform_sqrt=False, visualise=False, 
                        feature_vector=feature_vec)

            hog_features = np.concatenate((hog_features, features))

        hog_features = np.ravel(hog_features)   
        
        return hog_features

def test_hog_frature():
    # Read in image
    cars = glob.glob('train_data/vehicles/GTI_Far/*.png')
    # Generate a random index to look at a car image   
    car_ind = np.random.randint(0, len(cars))
    ind = np.random.randint(0, len(cars))
    # Read in the image
    print(cars[ind])
    rgb_image = mpimg.imread(cars[ind])    
    
    features, hog_image = get_hog_features(rgb_image, color_space='GRAY', vis=True, feature_vec=True)
    
    print(len(features), len(hog_image))
                        
    fig = plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.imshow(rgb_image)
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(hog_image[0])
    plt.title('HOG Image')
    fig.tight_layout()
    plt.show()
                     
# Define a function to compute color histogram features  
# Pass the color_space flag as 3-letter all caps string
# like 'HSV' or 'LUV' etc.
def bin_spatial(rgb_img, color_space='RGB', size=(16, 16)):
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
             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features
    

# Define a function to compute color histogram features  
def color_hist(rgb_img, color_space='RGB', nbins=128, bins_range=(0, 256)):
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
        
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(feature_image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:,:,2], bins=nbins, range=bins_range)
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
    

# Define a function to extract features from a list of images
def extract_features(rgb_image):
    # Apply hog() to get hog features
    hog_features_1 = get_hog_features(rgb_image, color_space='GRAY', vis=False, feature_vec=True)
    #hog_features_2 = get_hog_features(rgb_image, color_space='YUV', vis=False, feature_vec=True)
    #print("hog_features length:", len(hog_features_1))
    # Apply bin_spatial() to get spatial color features
    spatial_features_1 = bin_spatial(rgb_image, color_space='RGB')
    spatial_features_2 = bin_spatial(rgb_image, color_space='LUV')
    #print("spatial_features length:", len(spatial_features_1))
    # Apply color_hist() to get color histogram features
    hist_features_1 = color_hist(rgb_image, color_space='RGB')
    hist_features_2 = color_hist(rgb_image, color_space='YUV')
    #print("hist_features length:", len(hist_features_1))
    # Return list of feature vectors
    return np.concatenate((hog_features_1, spatial_features_1, spatial_features_2, hist_features_1, hist_features_2))


################################ Train Classify ################################
    
def train_classifier() :
    
    car_features = []
    notcar_features = []
    
    notcars = glob.glob('train_data/non-vehicles/Extras/*.png')
    for file_path in glob.glob('train_data/non-vehicles/GTI/*.png'):
        notcars.append(file_path)
    
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
        rgb_image = mpimg.imread(file_path)
        # Extract features
        features = extract_features(rgb_image)
        #print("feature length:", len(features))
        # Append the new feature vector to the features list
        car_features.append(features)
    # Iterate through the list of images
    for file_path in notcars:
        # Read in each one by one
        rgb_image = mpimg.imread(file_path)
        # Extract features
        features = extract_features(rgb_image)
        #print("feature length:", len(features))
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
    
    X, y = shuffle(X, y)
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    print('length of train features:', len(X_train))    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
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
    
    # Check the prediction time for a single sample
    t_start=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t_end = time.time()
    print(round(t_end-t_start, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    return svc, X_scaler

################################## Search Car ##################################
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    if x_start_stop[0] < 0 :
        x_start_stop[0] = 0
    if x_start_stop[1] > img.shape[1]:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] < 0:
        y_start_stop[0] = 0
    if y_start_stop[1] > img.shape[0]:
        y_start_stop[1] = img.shape[0]
        
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
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def generate_serach_windows(image) :
    search_window_list = [] 
    print("image size X:", image.shape[1])
    print("image size Y:", image.shape[0])
    for counter in range(10):
        window_size = 32 * (counter+1)
        windows = slide_window(image, x_start_stop=[0, image.shape[1]], 
                    y_start_stop=[image.shape[0]/2, image.shape[0]/2+120+counter*24], 
                    xy_window=(window_size, window_size), xy_overlap=(0.5, 0.5))
        for window in windows :
            search_window_list.append(window)       
    print(search_window_list[0])
    print("Number of search windows:", len(search_window_list))
    return search_window_list


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_cars(orignal_image, search_windows, clf, scaler):
    #1) Create an empty list to receive positive detection windows
    hit_windows = []
    #2) Iterate over all windows in the list
    for window in search_windows:
        #3) Extract the test window from original image
        feature_img = cv2.resize( orignal_image[
                                int(window[0][1]):int(window[1][1]), 
                                int(window[0][0]):int(window[1][0])], 
                                (64, 64) )      
        #4) Extract features for that window using single_img_features()
        features = extract_features(feature_img)
        #5) Scale extracted features to be fed to classifier
        #window_features = scaler.transform(np.array(features).reshape(1, -1))
        window_features = scaler.transform(features.astype(np.float64).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(window_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            hit_windows.append(window)
    #8) Return windows for positive detections
    return hit_windows


############################### Argument Display ###############################

def add_heat(orignal_image, hit_windows):
    heatmap = np.zeros_like(orignal_image[:,:,0]).astype(np.float)
    # Iterate through list of bboxes
    for box in hit_windows:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[int(box[0][1]):int(box[1][1]), int(box[0][0]):int(box[1][0])] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(image, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return image

def argument_display(orignal_image, hit_windows) :

    # Add heat to each box in box list
    heatmap = add_heat(orignal_image, hit_windows)
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, threshold=3)
    # Visualize the heatmap when displaying    
    #heatmap_visual = np.clip(heatmap, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    #print(labels[1], 'cars found')
    #plt.imshow(labels[0], cmap='gray')
    draw_img = draw_labeled_bboxes(np.copy(orignal_image), labels)
    
    return draw_img, heatmap

################################ Vehicle Detect ################################
def vehicle_detection_images(clf, scaler) :

    features = []
  
    # read in images
    images_path = glob.glob('test_images/*.jpg')
    print("Images number:", len(images_path))
    
    # Iterate through the list of images
    for file_path in images_path:
        # Read in each one by one
        rgb_image = mpimg.imread(file_path)
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        orignal_image = rgb_image.astype(np.float32)/255
        # process images
        search_windows = generate_serach_windows(orignal_image)
        hit_windows = search_cars(orignal_image, search_windows, clf, scaler)
        draw_img, heatmap = argument_display(orignal_image, hit_windows) 
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()
        plt.show()
    
def image_prcess(rgb_image, clf, scal):
    orignal_image = rgb_image.astype(np.float32)/255
    # process images
    search_windows = generate_serach_windows(orignal_image)
    hit_windows = search_cars(orignal_image, search_windows, clf, scaler)
    draw_img, heatmap = argument_display(orignal_image, hit_windows) 
    
def vehicle_detection_video(classfy, scaler) :
    pass

##################################### Run #####################################
def run() :
    #test_hog_frature()
    clf, scaler = train_classifier()
    #clf = None
    #vehicle_detection_images(clf, scaler)
    #vehicle_detection_video(clf, scaler)
    
    def image_prcess(rgb_image):
        orignal_image = rgb_image.astype(np.float32)/255
        # process images
        search_windows = generate_serach_windows(orignal_image)
        hit_windows = search_cars(orignal_image, search_windows, clf, scaler)
        draw_img, heatmap = argument_display(orignal_image, hit_windows) 
        return draw_img*255
    #read in video
    write_output = 'output_images/abc.mp4'

    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(image_prcess) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    
    
if __name__ == '__main__':
    run()   
    
