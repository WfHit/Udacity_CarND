#/usr/bin/env python3
'''
'''
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML

################################ Pipeline ######################################

############################ Image undistort ###################################
def image_undistort(image, mtx, dist) :
    return cv2.undistort(image, mtx, dist, None, mtx)
    
######################### Detect lane line edge ################################
def image_egde_detect(img, s_thresh=(170, 255), sx_thresh_gray=(20, 100), sx_thresh_l=(20, 100)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image and in RGB format
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    ######################## Detect edge using S channel #######################
    # Threshold S channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    ################# Detect edge using sobel x on gray image ##################
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx_gray = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx_gray = np.absolute(sobelx_gray) 
    # Scale iamge
    scaled_sobel_gray = np.uint8(255*abs_sobelx_gray /np.max(abs_sobelx_gray))
    # Threshold x gradient
    thresh_min_gray = sx_thresh_gray[0]
    thresh_max_gray = sx_thresh_gray[1]
    sxbinary_gray = np.zeros_like(scaled_sobel_gray)
    sxbinary_gray[(scaled_sobel_gray >= thresh_min_gray) & (scaled_sobel_gray <= thresh_max_gray)] = 1

    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    ################# Detect edge using sobel x on l_channel ###################
    # Sobel x
    sobelx_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx_l = np.absolute(sobelx_l) 
    # Scale iamge
    scaled_sobel_l = np.uint8(255*abs_sobelx_l /np.max(abs_sobelx_l))
    # Threshold x gradient
    thresh_min_l = sx_thresh_l[0]
    thresh_max_l = sx_thresh_l[1]
    sxbinary_l = np.zeros_like(scaled_sobel_l)
    sxbinary_l[(scaled_sobel_l >= thresh_min_l) & (scaled_sobel_l <= thresh_max_l)] = 1    
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((sxbinary_l, sxbinary_gray, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_binary == 1) | (sxbinary_l == 1) | (sxbinary_gray == 1)] = 1 # 
    
    return color_binary, combined_binary
    
########################### Image perspective ##################################
def image_perspective(binary_img):
    img_size = (binary_img.shape[1], binary_img.shape[0])  
    
    src = np.float32([[565, 460],
                      [715, 460],
                      [1150, 720],
                      [130, 720]])                    
    dst = np.float32([[440, 0],
                      [950,0],
                      [950,720],
                      [440, 720]])          
    '''
    cv2.line(binary_img, (565, 460), (725, 460), (0, 0, 255), thickness=10)  
    cv2.line(binary_img, (565, 460), (130, 720), (0, 0, 255), thickness=10)    
    cv2.line(binary_img, (1250, 720), (725, 460), (0, 0, 255), thickness=10)      
    '''   
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Mivs = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(binary_img, M, img_size)

    warped[ : , 0:400] = 0
    warped[ : , 1080:1280] = 0

    color_warped = warped * 255
    # Return the resulting image and matrix
    return warped, color_warped, M, Mivs
    
########################### Identify lane-line #################################

#----------------------------- Histogram way -----------------------------------
def identify_lane_line_histogram_single(binary_warped):
    # Assuming have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero_pionts = binary_warped.nonzero()
    nonzeroy = np.array(nonzero_pionts[0])
    nonzerox = np.array(nonzero_pionts[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) ).nonzero()[0]
        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high) ).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return leftx, lefty, rightx, righty

def identify_lane_line_histogram_video(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero_pionts = binary_warped.nonzero()
    nonzeroy = np.array(nonzero_pionts[0])
    nonzerox = np.array(nonzero_pionts[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & \
                            (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & \
                               (nonzerox < (right_fit[0]*(nonzeroy**2) +     right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return leftx, lefty, rightx, righty


#---------------------------- Convolution way ----------------------------------

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(image.shape[0]*3/4):, :int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(image.shape[0]*3/4):, int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height): int(image.shape[0]-level*window_height), : ], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids
 
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output
               
def identify_lane_line_convolution(binary_warped):
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin)

    leftx = lefty = rightx = righty = None

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
	        l_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
	        r_mask = window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)
	        # Add graphic points from window mask here to total pixels found 
	        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
	        
        nonzero_pionts_left = l_points.nonzero()
        leftx = np.array(nonzero_pionts_left[0])
        lefty = np.array(nonzero_pionts_left[1])

        nonzero_pionts_right = r_points.nonzero()
        rightx = np.array(nonzero_pionts_right[0])
        righty = np.array(nonzero_pionts_right[1])
        '''
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        ''' 
    return leftx, lefty, rightx, righty

    
###################### Calculate radius & position #############################
def cal_radius_postion(leftx, lefty, rightx, righty) :
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_left = np.max(lefty)
    y_eval_right = np.max(lefty)
    y_eval = np.min([y_eval_left, y_eval_right])
    print(y_eval)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    return left_curverad, right_curverad
    

############################# Argument display #################################
def argument_display(undist, warped, Mivs, leftx, lefty, rightx, righty) :

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval_left = np.max(lefty)
    y_eval_right = np.max(righty)
    y_eval = np.min([y_eval_left, y_eval_right])
    
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
   

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Mivs, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result

def process_image():
    with open('calibrateCamera.pkl', 'rb') as f:
        cal_pickle = pickle.load(f)
    mtx = cal_pickle["mtx"]
    dist = cal_pickle["dist"]
    
    images = glob.glob('test_images/test*.jpg')
    print(len(images))

    counter = 0
    for image_file_path in images:
        # Read in an image
        image = cv2.imread(image_file_path)
        
        undist = cv2.undistort(image, mtx, dist, None, mtx)
        #image = image.squeeze()
        #undist = undist.squeeze()
        cv2.imwrite('output_images/test_images/undist_'+image_file_path.split('/')[-1], undist)
        
        color_binary, combined_binary = image_egde_detect(undist)
        cv2.imwrite('output_images/test_images/binary_'+image_file_path.split('/')[-1], color_binary)

        warped, color_warped, M, Mivs = image_perspective(combined_binary)
        cv2.imwrite('output_images/test_images/warped_'+image_file_path.split('/')[-1], color_warped)
        
        leftx, lefty, rightx, righty = identify_lane_line_histogram_single(warped)
        #leftx, lefty, rightx, righty = identify_lane_line_convolution(warped)
        
        result = argument_display(undist, warped, Mivs, leftx, lefty, rightx, righty)
        cv2.imwrite('output_images/test_images/result_'+image_file_path.split('/')[-1], result)
  

with open('calibrateCamera.pkl', 'rb') as f:
    cal_pickle = pickle.load(f)
mtx = cal_pickle["mtx"]
dist = cal_pickle["dist"]
          
def process_image_1(image):

    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    color_binary, combined_binary = image_egde_detect(undist)

    warped, color_warped, M, Mivs = image_perspective(combined_binary)
    
    leftx, lefty, rightx, righty = identify_lane_line_histogram_single(warped)
    
    result = argument_display(undist, warped, Mivs, leftx, lefty, rightx, righty)

    return  result
              
def process_video():
        
    write_output = 'output_images/abc.mp4'

    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image_1) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    
#process_image()

process_video()
    
