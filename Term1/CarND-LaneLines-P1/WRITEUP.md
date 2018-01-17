# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

**The goals / steps of this project are the following:** 
* Make a pipeline that finds lane lines on the road 
* Reflect on work in a written report 


[//]: # (Image References)

[grayscale_image]: ./writeup_images/solidWhiteRight/solidWhiteRight_gray.jpg "Grayscale"
[blur_image]: ./writeup_images/solidWhiteRight/solidWhiteRight_blur.jpg "Gaussian smoothing"
[canny_image]: ./writeup_images/solidWhiteRight/solidWhiteRight_canny_edges.jpg "Canny Edge Detection"
[masked_image]: ./writeup_images/solidWhiteRight/solidWhiteRight_roi.jpg "Region of Interest"
[line_image]: ./writeup_images/solidWhiteRight/solidWhiteRight_line.jpg "Hough Transform"
[final_image]: ./test_images_output/solidWhiteRight.jpg "Final image"

---

## Reflection
### 1. Describe my pipeline. As part of the description, explain how I modified the draw_lines() function.
**My pipeline consisted of 6 steps:**  
1. I converted the images to grayscale.
![alt text][grayscale_image]  
2. I used Gaussian smoothing to reduce noise in the images.
![alt text][blur_image]  
3. I using the Canny image detection algorithm to find boundaries in the images.
![alt text][canny_image]  
4. I used a polyline to isolate the region of interest (trapezoid). I had to do some fine tuning to find a good compromise here.
![alt text][masked_image]  
5. I uses the Hough Trasform to identify the lines in the Region of Interest.
![alt text][line_image]  
6. I stacked the image with lines on top of the original image.
![alt text][final_image]  

**In order to draw a single line on the left and right lanes, I modified the draw_lines() function for doing the following:**
1. For each line calculates its slope 
2. According to the slope value, put the line piont to left line lsit or right line list
3. Using the value in the line list to fit a line
4. Calculate the up and down boundaries of line to draw
5. Draw the lines

### 2. Identify potential shortcomings with your current pipeline

While evaluating the result of my algorithm I identified the following issues: 
* Handling dashed lines: I had to do some fine tuning to get something acceptable 
* Handling light changes in the image and color changes in the road (issue visible by looking at the third video) 
* Handling slope changes and curves in the road 

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to process the image in order to have better robustness agains light changes in the image, for instance increasing the contrast. Another potential improvement could be to process the image in such a way to isolate and extract yellow and white lines. 
