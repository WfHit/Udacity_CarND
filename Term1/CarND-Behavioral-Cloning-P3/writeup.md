# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[Original_Image]: ./writeup_images/original_image.jpg "Original Image"
[Flipped_Image]: ./writeup_images/image_flip.jpg "Flipped Image"
[TrainLoss_Image]: ./writeup_images/TrainLoss_Image.png "TrainLoss_Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 track one video capture

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. A generator is defined in function 'create_data_generator' from line 68 to line 105.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The first layer of my network is a Cropping2D layer which cut out top 70 rows and bottom 20 rows pixels of image(model.py lines 115) , then followed by a Lambda layer to normalize data(model.py lines 117) . And the next is 5 Conv2D layers with first three layers using 5x5 filter size and last two layers using 3x3 filter size, and all 5 Conv2D layers using RELU activation to intoduce nonlinearity(model.py lines 119-130) . After the 5 Conv2D layers is a Flatten layer(model.py lines 132). Last, 4 Fully Connected layer are added to the model after Flatten layer with RELU activation(model.py lines 134-146) . 

#### 2. Attempts to reduce overfitting in the model

With the last 5 Fully Connected layer in the model, a dropout layer is added after each of the front four layer to reduce overfitting (model.py lines 135 138 141)

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 32-65). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I use the data provided by udacity, both left and right camera datas are added to training datas. 

### Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reference an exist architecture and modify it to fit the requirement.

My first step was to use a convolution neural network model similar to the DAVE-2 I thought this model might be appropriate because nvidia designs it to slove this kind of problem on purpose.

To build this model, first I read the parper which discribes the architecture, then I build a network with a Lambda layer to normalize data, 5 Conv2D layers using RELU, 5 Fully Connected layer using RELU and four dropout layer to reduce overfitting. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model can not run, beacuse it just consume all the resource of my GPU, so I add a Cropping2D layer which cut out top 70 rows and bottom 20 rows pixels of image on the one hand to reduce consumed resource and on the other hand to avoid unnecessary compution. After that it works well except that the car will leave the road at some turning place, so I increase the left and right camera compensation angle to 0.5, and slove this problem.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 108-145) consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        | 
|:---------------------:|:-------------------------:| 
| Cropping           	|                           |
| Normalization         |                           |
| Convolution 5x5     	| 1x1 stride, valid padding |
| RELU					|							|
| Max pooling	      	| 2x2 stride, valid padding |
| Convolution 5x5     	| 1x1 stride, valid padding |
| RELU					|							|
| Max pooling	      	| 2x2 stride, valid padding |
| Convolution 5x5     	| 1x1 stride, valid padding |
| RELU					|							|
| Max pooling	      	| 2x2 stride, valid padding |
| Convolution 3x3     	| 1x1 stride, valid padding |
| RELU					|							|
| Convolution 3x3     	| 1x1 stride, valid padding |
| RELU					|							|
| Flatten	            |   		                |
| Fully connected		|   	                    |
| RELU					|							|
| Dropout 0.5			|							|
| Fully connected		|                           |
| RELU					|							|
| Dropout 0.5			|							|
| Fully connected		|   		                |
| RELU					|							|
| Dropout 0.5			|							|
| Fully connected		|   	                    |
| Linear				|							|

#### 3. Creation of the Training Set & Training Process

I use the data provided by udacity. Here is an example image of center lane driving:

![alt text][Original_Image]

To augment the data sat, I also flipped images and angles thinking that this would add data that represents driving inversely. For example, here is an image that has then been flipped:

![alt text][Flipped_Image]

After the collection process, I had 38400 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as loss decreases very slow after 5 loops as show in below picture. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][TrainLoss_Image]

### Reference
[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

