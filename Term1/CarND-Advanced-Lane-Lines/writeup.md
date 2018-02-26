## Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_test]: ./writeup_images/undistorted_test.png "undistorted_test"
[undistorted_road]: ./writeup_images/undistorted_road.jpg "undistorted_road"
[binary_road]: ./writeup_images/binary_road.jpg "binary_road"
[warped_road]: ./writeup_images/warped_road.jpg "warped_road"
[result_road]: ./writeup_images/result_road.jpg "result_road"
[project_video]: ./writeup_images/project_video.mp4 "project_video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called "CalibrateCamera.py".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp_corners` is just a replicated array of coordinates (CalibrateCamera.py lines 27-28), and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image (CalibrateCamera.py lines 47).  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection (CalibrateCamera.py lines 46).  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function (CalibrateCamera.py lines 53).  I applied this distortion correction to the test image using the `cv2.undistort()` function (CalibrateCamera.py lines 74) and obtained this result: 

![alt text][undistorted_test]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistorted_road]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 124 through 283 in `AdvancedLaneLines.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][binary_road]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `image_perspective()`, which appears in lines 386 through 305 in the file `AdvancedLaneLines.py`. I chose the hardcode the source and destination points in the following manner:

```
    src = np.float32([[566, 460],
                      [715, 460],
                      [1150, 720],
                      [130, 720]])                    
    dst = np.float32([[440, 0],
                      [840,0],
                      [840,720],
                      [440, 720]])  
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 566, 460      | 440, 0        | 
| 715, 460      | 840, 0        |
| 1150, 720     | 840, 720      |
| 130, 720      | 440, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warped_road]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identified lane-line pixels includes a function called `hist_detect_lane()` and `hist_detect_lane_with_filter()`, which appears in lines 308 through 374 and 377 through 439 in the file `AdvancedLaneLines.py`. The difference between this two function is that the last one using a foward filter. 

The code for fit their positions with a polynomial includes a function called `cal_parabola_pixels()`, which appears in lines 388 through 410 in the file `AdvancedLaneLines.py`. It return the fitted parabola coefficients in pixels. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculated the radius of curvature of the lane in lines 742 through 746, included in a function called `process_image()`, which appears in the file `AdvancedLaneLines.py`.

The code for calculated the vehicle includes a function called `cal_car_offset()`, which appears in lines 610 through 621 in the file `AdvancedLaneLines.py`. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 555 through 588 in my code in `AdvancedLaneLines.py` in the function `augmented_display()`.  Here is an example of my result on a test image:

![alt text][result_road]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result in folder ./writeup_images](./writeup_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During tuning my pipeline, I find it is likely to fail when the cars passing to light colour road. I think I need figure out a better method to identify the lane line when its color close to road's color, for example using a relative threshold rather than a fix one.
