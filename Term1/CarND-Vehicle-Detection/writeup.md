## Writeup Report

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]: ./writeup_images/cars.png
[not_cars]: ./writeup_images/not_cars.png
[cars_hog_image_h]: ./writeup_images/cars_hog_image_h.png
[cars_hog_image_l]: ./writeup_images/cars_hog_image_l.png
[cars_hog_image_s]: ./writeup_images/cars_hog_image_s.png
[not_cars_hog_image_h]: ./writeup_images/not_cars_hog_image_h.png
[not_cars_hog_image_l]: ./writeup_images/not_cars_hog_image_l.png
[not_cars_hog_image_s]: ./writeup_images/not_cars_hog_image_s.png
[result]: ./writeup_images/result.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 24 through 91 of the file called `VehicleDectectionSVM.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][cars]
![alt text][not_cars]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][cars_hog_image_h]
![alt text][cars_hog_image_l]
![alt text][cars_hog_image_s]
![alt text][not_cars_hog_image_h]
![alt text][not_cars_hog_image_l]
![alt text][not_cars_hog_image_s]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally I choose HOG parameters as follow:

| Parameters        | Value         | 
|:-----------------:|:-------------:| 
| color             | YCrCb         | 
| orientations      | 11            |
| pixels_per_cell   | (8, 8)        |
| cells_per_block   | (2, 2)        |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the combination of HOG features and color features which in lines 182 through 281 of the file called `VehicleDectectionSVM.py`. 
First I read in all training images and which are more than 8000 for each of cars and not_cars, but I used the front 8000 of both cars and not_cars images for easily compute data size. 
Then I extract all the features from images, shuffle it, and split to train set and valid set with ratio of 0.2, and normalize the data with `StandardScaler()`
Next I train the SVM with the normalized data and save the result to file `clf_pickle.pkl`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implement sliding window search in lines 285 through 461 of the file called `VehicleDectectionSVM.py`. And parameters about search windows as follow:
    
| Parameters    | Value                                     | 
|:-------------:|:-----------------------------------------:| 
| overlap       | (0.7, 0.7)                                | 
| scales        | (64,64),      (128,128),    (256,256)     |
| x_start_stop  | (None, None), (None, None), (None, None)  |
| y_start_stop  | (400, 540),   (380, 650),   (360, None)   |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][result]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./writeup_images/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After a lot of struggle on SVM, I just find out that using SVM to detect cars is time consuming and unstable, and I could not get an acceptable result. So I have to turn to SSD. 
Here is how I use SSD to detect cars in video which in file `VehicleDectectionSSD.py`. First I use a trained SSD to detect cars on single image, then I created a heatmap and thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issues I faced is that though the trained SVM working perfect on test data, it performance is not acceptable on video frame, I'm not sure why this appends, may be the model is overfited. And it is more accuracy to detect black car than detect white cars, especially when cars is far and its image is small. 
When I used the SVM, pipeline is likely to fail when the cars pass the white bridge using SVM, I think I need to add some features in other color space to decrease the influence of the road background color.

### Reference 

[SSD Model](https://github.com/rykov8/ssd_keras)

