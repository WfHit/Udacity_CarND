# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[data_set_exploration]: ./writeup_images/data_set_exploration.png "data_set_exploration"
[data_set_histogram_Train]: ./writeup_images/data_set_histogram_Train.png "data_set_histogram_Train"
[data_set_histogram_Vaild]: ./writeup_images/data_set_histogram_Vaild.png "data_set_histogram_Vaild"
[data_set_histogram_Test]: ./writeup_images/data_set_histogram_Test.png "data_set_histogram_Test"
[image_preprocess]: ./writeup_images/image_preprocess.png "image_preprocess"
[new_test_images]: ./writeup_images/new_test_images.png "new_test_images"
[new_test_images_top_five]: ./writeup_images/new_test_images_top_five.png "new_test_images_top_five"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/WfHit/Udacity_CarND/tree/master/Term1/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributes.
It is obvious that the distribution of the three sets is the same, but the frequency of traffic signs in each set is different.
![alt text][data_set_exploration]
![alt text][data_set_histogram_Train]
![alt text][data_set_histogram_Vaild]
![alt text][data_set_histogram_Test]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in this project the most critical factor in identifying traffic signs is gradients. Gradients mean edges. This is the most essential part, while computing gradients naturally use grayscale images. The color itself is very easy to be influenced by light and other factors, and as a result, the color of the traffic signs of the same kind has a lot of change. So the color itself is difficult to provide key information. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image_preprocess]

As a last step, I normalized the image data because we need to regulate the value of each dimension of the data, makes the data vector eventually falls in the range of [0,1] or [-1,1]. This is important for subsequent processing, because many default parameters, assume that the data has been scaled to a reasonable interval.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten	            |            				 outputs 400		|
| Fully connected		|         				     outputs 120		|
| RELU					|												|
| Dropout 0.5			|												|
| Fully connected		|         				     outputs 84		    |
| RELU					|												|
| Dropout 0.5			|												|
| Fully connected		|         				     outputs 43		    |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following parameters

| Items         		|     Vaule     	        					| 
|:---------------------:|:---------------------------------------------:| 
| optimizer        		| AdamOptimizer       							| 
| batch size        	| 200                                        	|
| epochs				| 30    										|
| learning rate	      	| 0.001                                         |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.952
* test set accuracy of 0.938

Why I think the architecture is suitable for the current problem:
* I choose to use a well known architecture: LeNet-5.
* Because, as we all known, LeNet is a very classic structure, it has a simple structure and easy to training, LeNet has been shown that can be used for recognition of handwritten font. On the other hand, traffic sign images are similar to handwritten font images both in image size and complexity of context, so I think that LeNet is perfect choose to slove the task of recognition traffic signs.
* As shown about, the accuracies on training, validation and test are both more than 93%, this is the evidence that the model is working well.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new_test_images] 

The fourth image might be difficult to classify because the picture is dim and hard to be identified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit(20km/h)   | Speed limit(20km/h)  							| 
| Speed limit(30km/h)   | Speed limit(30km/h)  							|
| Traffic signals		| Traffic signals								|
| Wild animals crossing | Wild animals crossing					 		|
| Keep left			    | Keep left     							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.938

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


![alt text][new_test_images_top_five] 


For the first image, the model is relatively sure that this is a Speed limit(20km/h) sign , and the image does contain a Speed limit(20km/h) sign. The top five soft max probabilities were shown in up graph.

For the second image, the model is relatively sure that this is a Speed limit(30km/h) sign , and the image does contain a Speed limit(30km/h) sign. The top five soft max probabilities were shown in up graph.

For the third image, the model is relatively sure that this is a Traffic signals sign , and the image does contain a Traffic signals sign. The top five soft max probabilities were shown in up graph.

For the fourth image, the model is relatively sure that this is a Wild animals crossing sign , and the image does contain a Wild animals crossing sign. The top five soft max probabilities were shown in up graph.

For the fifth image, the model is relatively sure that this is a Keep left sign , and the image does contain aKeep left sign. The top five soft max probabilities were shown in up graph.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


