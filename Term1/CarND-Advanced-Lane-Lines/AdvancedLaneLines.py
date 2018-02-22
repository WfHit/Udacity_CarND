#/usr/bin/env python3
'''
'''
import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# Define a class to receive the characteristics of each line detection
class C_LaneLine_t:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #camera distortion mtx
        self.camera_mtx = None
        #camera distortion dist
        self.camera_dist = None
        #threshold for h_channel
        self.hls_h_channel_thresh = [15, 50]
        #threshold for s_channel
        self.hls_s_channel_thresh = [170, 255]
        #threshold for b_channel
        self.lab_b_channel_thresh = [190, 255]
        #threshold for sobel x on gray image
        self.sobelx_gray_thresh = [30, 255]
        #threshold for sobel x on gray image
        self.sobely_gray_thresh = [60, 255]
        #threshold for sobel x on l channel image
        self.sobelx_hls_l_channel_thresh = [30, 100]
        #threshold for sobel x y mag on gray image
        self.sobelxy_mag_gray_thresh = [90, 255]
        #threshold for sobel x y dir on gray image
        self.sobelxy_dir_gray_thresh = [0.8, 1.25]
        #threshold for r channel on rgb image
        self.rgb_r_channel_thresh = [200, 255]
        #perspective transform matrix
        self.perspect_matrix = None
        #inverse perspective transform matrix
        self.ivs_perspect_matrix = None
        #meters per pixel in y dimension
        self.ym_per_pix = 30/720 
        #meters per pixel in x dimension
        self.xm_per_pix = 3.7/700 
        #left fited parabola coefficients list
        self.left_fit_list = []
        #right fited parabola coefficients list
        self.right_fit_list = []
        #failed_detected_counter
        self.failed_detected_counter = 0
        #pipeline is working
        self.working  = False
        #pipeline need restart
        self.need_reset = False
        #current best left fit
        self.left_fit = None
        #current best right fit
        self.right_fit = None
        #restart counter
        self.restart_counter = 0
        #check fail counter
        self.check_fail_counter = 0
        
    def load_camera_parameter(self):
        with open('calibrateCamera.pkl', 'rb') as f:
            cal_pickle = pickle.load(f)
        self.camera_mtx = cal_pickle["mtx"]
        self.camera_dist = cal_pickle["dist"]
    
    
    def calc_m_mivs(self, src, dst):
        # Given src and dst points, calculate the perspective transform matrix
        self.perspect_matrix = cv2.getPerspectiveTransform(src, dst)
        self.ivs_perspect_matrix = cv2.getPerspectiveTransform(dst, src)   
        #return self.perspect_matrix, self.ivs_perspect_matrix    
    
    
    def image_undistort(self, rgb_image):
        '''
        image undistortion
        '''
        if (self.camera_mtx == None) or (self.camera_dist == None) :
            print("camera parameter no initial yet")
        undist_image = cv2.undistort(rgb_image, self.camera_mtx, self.camera_dist, None, self.camera_mtx)
        #mpimg.imsave('output_images/test_images/undist_'+image_file_path.split('/')[-1], undist_image)
        return undist_image

                   
    def highlight_lane_line(self, rgb_image):
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image and in RGB format
        
        # Find lane line on EGB color space
        rgb_image_r_channel = rgb_image[:,:,0]
        rgb_image_g_channel = rgb_image[:,:,1]
        rgb_image_v_channel = rgb_image[:,:,2]
        
        # Detect lane line using R channel
        thresh_min = self.rgb_r_channel_thresh[0]
        thresh_max = self.rgb_r_channel_thresh[1]
        rgb_r_channel_binary_image = np.zeros_like(rgb_image_r_channel)
        rgb_r_channel_binary_image[(rgb_image_r_channel >= thresh_min) & (rgb_image_r_channel <= thresh_max)] = 1
    
        # Find lane line on HLS color space
        hls_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
        hls_image_h_channel = hls_image[:,:,0]
        hls_image_l_channel = hls_image[:,:,1]
        hls_image_s_channel = hls_image[:,:,2] 
        
        # Detect lane line using S channel
        thresh_min = self.hls_s_channel_thresh[0]
        thresh_max = self.hls_s_channel_thresh[1]
        hls_s_channel_binary_image = np.zeros_like(hls_image_s_channel)
        hls_s_channel_binary_image[(hls_image_s_channel >= thresh_min) & (hls_image_s_channel <= thresh_max)] = 1
        
        # Detect lane line using H channel
        thresh_min = self.hls_h_channel_thresh[0]
        thresh_max = self.hls_h_channel_thresh[1]
        hls_h_channel_binary_image = np.zeros_like(hls_image_h_channel)
        hls_h_channel_binary_image[(hls_image_h_channel >= thresh_min) & (hls_image_h_channel <= thresh_max)] = 1
        
        # Combine h & s channel 
        #hs_channel_binary_image = np.zeros_like(image_s_channel)
        #hs_channel_binary_image[(h_channel_binary_image == 1) | (s_channel_binary_image == 1)] = 1 
        
        # Detect lane line using sobel x on hls_l_channel_image       
        sobelx_hls_l_channel_image = cv2.Sobel(hls_image_l_channel, cv2.CV_64F, 1, 0)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx_hls_l_channel_image = np.absolute(sobelx_hls_l_channel_image) 
        # Scale iamge
        scaled_sobelx_hls_l_channel_image = np.uint8(255*abs_sobelx_hls_l_channel_image /np.max(abs_sobelx_hls_l_channel_image))
        # Threshold x gradient
        thresh_min = self.sobelx_hls_l_channel_thresh[0]
        thresh_max = self.sobelx_hls_l_channel_thresh[1]
        sobelx_binary_hls_l_channel_image = np.zeros_like(scaled_sobelx_hls_l_channel_image)
        sobelx_binary_hls_l_channel_image[(scaled_sobelx_hls_l_channel_image >= thresh_min) & (scaled_sobelx_hls_l_channel_image <= thresh_max)] = 1  
        
        # Find lane line on gray image 
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)     
        # Detect lane line using sobel x on gray image
        # Sobel x
        sobelx_gray_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx_gray_image = np.absolute(sobelx_gray_image) 
        # Scale iamge
        scaled_sobelx_gray_image = np.uint8(255*abs_sobelx_gray_image / np.max(abs_sobelx_gray_image))
        # Threshold x gradient
        thresh_min = self.sobelx_gray_thresh[0]
        thresh_max = self.sobelx_gray_thresh[1]
        sobelx_binary_gray_image = np.zeros_like(scaled_sobelx_gray_image)
        sobelx_binary_gray_image[(scaled_sobelx_gray_image >= thresh_min) & (scaled_sobelx_gray_image <= thresh_max)] = 1        
        # Sobel y
        sobely_gray_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobely_gray_image = np.absolute(sobely_gray_image) 
        # Scale iamge
        scaled_sobely_gray_image = np.uint8(255*abs_sobely_gray_image / np.max(abs_sobely_gray_image))
        # Threshold x gradient
        thresh_min = self.sobely_gray_thresh[0]
        thresh_max = self.sobely_gray_thresh[1]
        sobely_binary_gray_image = np.zeros_like(scaled_sobely_gray_image)
        sobely_binary_gray_image[(scaled_sobely_gray_image >= thresh_min) & (scaled_sobely_gray_image <= thresh_max)] = 1        
        # Sobel xy mag
        thresh_min = self.sobelxy_mag_gray_thresh[0]
        thresh_max = self.sobelxy_mag_gray_thresh[1]
        abs_sobelxy_mag_gray_image = np.sqrt(abs_sobelx_gray_image ** 2 + abs_sobely_gray_image ** 2)
        scaled_sobelxy_mag_gray_image = np.uint8(255*abs_sobelxy_mag_gray_image / np.max(abs_sobelxy_mag_gray_image))
        sobelxy_binary_mag_gray_image = np.zeros_like(scaled_sobelxy_mag_gray_image)
        sobelxy_binary_mag_gray_image[(scaled_sobelxy_mag_gray_image >= thresh_min) & (scaled_sobelxy_mag_gray_image <= thresh_max)] = 1
        # Sobel xy dir
        thresh_min = self.sobelxy_dir_gray_thresh[0]
        thresh_max = self.sobelxy_dir_gray_thresh[1]
        abs_sobelxy_dir_gray_image = np.arctan2(abs_sobelx_gray_image, abs_sobely_gray_image)
        scaled_sobelxy_dir_gray_image = np.uint8(255*abs_sobelxy_mag_gray_image / np.max(abs_sobelxy_mag_gray_image))
        sobelxy_binary_dir_gray_image = np.zeros_like(scaled_sobelxy_dir_gray_image)
        sobelxy_binary_dir_gray_image[(scaled_sobelxy_dir_gray_image >= thresh_min) & (scaled_sobelxy_dir_gray_image <= thresh_max)] = 1
    
        # Find lane line on LAB color space 
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
        lab_image_l_channel = hls_image[:,:,0]
        lab_image_a_channel = hls_image[:,:,1]
        lab_image_b_channel = hls_image[:,:,2] 
        
        thresh_min = self.lab_b_channel_thresh[0]
        thresh_max = self.lab_b_channel_thresh[1]
        lab_b_channel_binary_image = np.zeros_like(lab_image_b_channel)
        lab_b_channel_binary_image[(lab_image_b_channel >= thresh_min) & (lab_image_b_channel <= thresh_max)] = 1

        # Combine the three binary thresholds
        combined_binary = np.zeros_like(hls_image_s_channel)
        #combined_binary[(h_channel_binary_image == 1) ] = 1 
        combined_binary[((sobelx_binary_gray_image == 1) & (sobely_binary_gray_image == 1)) | \
                    ((hls_s_channel_binary_image == 1) & (hls_h_channel_binary_image == 1)) | \
                    (lab_b_channel_binary_image == 1) ] = 1 
        # Stack each channel to view their individual contributions in red, green and blue respectively
        # This returns a stack of the three binary images, whose components you can see as different colors
        color_binary = np.dstack((hls_s_channel_binary_image, sobelx_binary_gray_image, lab_b_channel_binary_image)) * 255 
        
        return combined_binary, color_binary

    
    def image_perspective(self, binary_img):
        img_size = (binary_img.shape[1], binary_img.shape[0])  
        #cv2.line(binary_img, (565, 460), (725, 460), (0, 0, 255), thickness=10)  
        #cv2.line(binary_img, (565, 460), (130, 720), (0, 0, 255), thickness=10)    
        #cv2.line(binary_img, (1250, 720), (725, 460), (0, 0, 255), thickness=10)         
        # Warp the image using OpenCV warpPerspective()
        warped_image = cv2.warpPerspective(binary_img, self.perspect_matrix, img_size)
        warped_image[ : , 0:400] = 0
        warped_image[ : , 1080:1280] = 0
        color_warped_image = np.dstack((warped_image, warped_image, warped_image)) * 255
        #color_warped_image = warped_image * 255
        # Return the resulting image and matrix
        return warped_image, color_warped_image         
        
        
    def hist_detect_lane(self, binary_warped):
        # Assuming have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)   
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[ : midpoint])
        rightx_base = np.argmax(histogram[midpoint : ]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 12
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
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
       
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
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        return (leftx, lefty, rightx, righty)
  
  
    def hist_detect_lane_with_filter(self, binary_warped, left_fit, right_fit):
        
        image_size_x_y = (binary_warped.shape[1], binary_warped.shape[0]) 

        lift_line_postion = left_fit[0]*image_size_x_y[1]**2 + left_fit[1]*image_size_x_y[1] + left_fit[2]
        right_line_postion = right_fit[0]*image_size_x_y[1]**2 + right_fit[1]*image_size_x_y[1] + right_fit[2]
        leftx_current = lift_line_postion
        rightx_current = right_line_postion
        
        # Choose the number of sliding windows
        nwindows = 12
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_pionts = binary_warped.nonzero()
        nonzeroy = np.array(nonzero_pionts[0])
        nonzerox = np.array(nonzero_pionts[1])
        
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
       
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
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        return (leftx, lefty, rightx, righty)
    
    def find_window_centroids(self, image, window_width, window_height, margin):
    
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
 
    def window_mask(self, width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output
               
    def identify_lane_line_convolution(self, binary_warped):
        # window settings
        window_width = 50 
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching

        window_centroids = self.find_window_centroids(binary_warped, window_width, window_height, margin)

        leftx = lefty = rightx = righty = None

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(binary_warped)
            r_points = np.zeros_like(binary_warped)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
	            l_mask = self.window_mask(window_width, window_height, binary_warped, window_centroids[level][0], level)
	            r_mask = self.window_mask(window_width, window_height, binary_warped, window_centroids[level][1], level)
	            # Add graphic points from window mask here to total pixels found 
	            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
	            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            nonzero_pionts_left = l_points.nonzero()
            lefty = np.array(nonzero_pionts_left[0])
            leftx = np.array(nonzero_pionts_left[1])

            nonzero_pionts_right = r_points.nonzero()
            righty = np.array(nonzero_pionts_right[0])
            rightx = np.array(nonzero_pionts_right[1])

        return (leftx, lefty, rightx, righty)
    
    
    def convert_to_meter(self, parabola_coefficients) :
        '''
        Once the parabola coefficients are obtained, in pixels, convert them into meters. 
        For example, if the parabola is x_p= a_p*(y_p**2) +b_p*y_p+c_p; 
        and mx and my are the scale for the x and y axis, respectively (in meters/pixel); 
        then the scaled parabola is x_m= mx/(my**2)*a*(y_m**2) + (mx/my)*b*y_m + mx*c                
        '''
        return (self.xm_per_pix / self.ym_per_pix**2 * parabola_coefficients[0], self.xm_per_pix / self.ym_per_pix * parabola_coefficients[1], self.xm_per_pix * parabola_coefficients[2] )
    
    def cal_parabola_pixels(self, image_size_x_y, left_right_pixels) :
        
        leftx = left_right_pixels[0]
        lefty = left_right_pixels[1]
        rightx = left_right_pixels[2]
        righty = left_right_pixels[3]

        if (len(leftx)>0) and (len(lefty)>0) and (len(rightx)>0) and (len(righty)>0) : 
            
            self.detected = True
            left_fit = np.polyfit(lefty, leftx, 2) 
            right_fit = np.polyfit(righty, rightx, 2)
            return left_fit, right_fit
        else : 
            self.detected = False
            left_fit = ()
            right_fit = ()
            return left_fit, right_fit
            
    def lane_line_postprocess(self, image_size_x_y, left_fit, right_fit) :
        #Sanity Check
        '''
        Checking that they have similar curvature
        Checking that they are separated by approximately the right distance horizontally
        Checking that they are roughly parallel
        '''
        if self.detected :
            if self.working :
                left_fit_meter = self.convert_to_meter(left_fit)
                right_fit_meter = self.convert_to_meter(right_fit)
                # Calculate the new rad of curvature
                left_curverad = ((1 + (2*left_fit_meter[0]*image_size_x_y[1]*self.ym_per_pix + left_fit_meter[1])**2)**1.5) / np.absolute(2*left_fit_meter[0])
                right_curverad = ((1 + (2*right_fit_meter[0]*image_size_x_y[1]*self.ym_per_pix + right_fit_meter[1])**2)**1.5) / np.absolute(2*right_fit_meter[0])
                if (np.absolute(left_curverad-right_curverad) < 1000) and (np.absolute((right_fit_meter[2]-self.lane_line_center) - (self.lane_line_center-left_fit_meter[2]))<1) :
                    if len(self.left_fit_list) < 10 :
                        self.left_fit_list.append(left_fit)
                    else :
                        temp_list = self.left_fit_list[1:]
                        temp_list.append(left_fit)
                        self.left_fit_list = temp_list
                    if len(self.right_fit_list) < 10 :
                        self.right_fit_list.append(right_fit)
                    else :
                        temp_list = self.right_fit_list[1:]
                        temp_list.append(right_fit)
                        self.right_fit_list = temp_list       
                else :
                    self.check_fail_counter += 1
                    self.detected  = False
            else :
                self.working = True
                left_fit_meter = self.convert_to_meter(left_fit)
                right_fit_meter = self.convert_to_meter(right_fit)
                # Calculate the new rad of curvature
                left_curverad = ((1 + (2*left_fit_meter[0]*image_size_x_y[1]*self.ym_per_pix + left_fit_meter[1])**2)**1.5) / np.absolute(2*left_fit_meter[0])
                right_curverad = ((1 + (2*right_fit_meter[0]*image_size_x_y[1]*self.ym_per_pix + right_fit_meter[1])**2)**1.5) / np.absolute(2*right_fit_meter[0])
                #self.left_fit = left_fit
                #self.right_fit = right_fit
                #self.lane_line_center = (left_fit_meter[2] + right_fit_meter[2]) / 2.0
                self.left_fit_list = []
                self.right_fit_list = []
                self.left_fit_list.append(left_fit)
                self.right_fit_list.append(right_fit)
                if (np.absolute(left_curverad-right_curverad) < 1500) :
                    self.detected = False
                    
        #Reset
        if not self.detected :
            self.failed_detected_counter += 1
            if self.failed_detected_counter > 6 : 
                self.failed_detected_counter = 0
                self.working  = False
                self.need_reset = True
                self.restart_counter += 1
        else :
            self.failed_detected_counter = 0
            self.need_reset = False
            self.check_fail_counter = 0
            
        print('self.failed_detected_counter', self.failed_detected_counter)
        print('self.working', self.working)
        print('self.need_reset', self.need_reset)
        print('len(self.left_fit_list)', len(self.left_fit_list))
        #print(self.left_fit_list)
        print('len(self.right_fit_list)', len(self.right_fit_list))
        #print(self.right_fit_list)    
        print('self.restart_counter', self.restart_counter)
        print('self.check_fail_counter', self.check_fail_counter)
        
        left_fit_mean = None
        right_fit_mean = None
        left_fit_meter_mean = None
        right_fit_meter_mean = None
        left_curverad = None
        right_curverad = None 
        
        if (not self.working) :
            return left_fit_mean, right_fit_mean, left_fit_meter_mean, right_fit_meter_mean, left_curverad, right_curverad    
              
        #Smoothing
        if (len(self.left_fit_list)>0) and (len(self.right_fit_list)>0) :
            fit_0_sum = 0.0
            fit_1_sum = 0.0
            fit_2_sum = 0.0
            for left_fit in self.left_fit_list :
                fit_0_sum += left_fit[0]
                fit_1_sum += left_fit[1]
                fit_2_sum += left_fit[2]
            left_fit_mean = (fit_0_sum/len(self.left_fit_list), fit_1_sum/len(self.left_fit_list), fit_2_sum/len(self.left_fit_list)) 
            fit_0_sum = 0.0
            fit_1_sum = 0.0
            fit_2_sum = 0.0
            for right_fit in self.right_fit_list :
                fit_0_sum += right_fit[0]
                fit_1_sum += right_fit[1]
                fit_2_sum += right_fit[2]
            right_fit_mean = (fit_0_sum/len(self.right_fit_list), fit_1_sum/len(self.right_fit_list), fit_2_sum/len(self.right_fit_list)) 
        
        self.left_fit = left_fit_mean
        self.right_fit = right_fit_mean
        
        left_fit_meter_mean = self.convert_to_meter(left_fit_mean)
        right_fit_meter_mean = self.convert_to_meter(right_fit_mean)
        
        self.lane_line_center = (left_fit_meter_mean[2] + right_fit_meter_mean[2]) / 2.0
        
        # Calculate the new rad of curvature
        left_curverad = ((1 + (2*left_fit_meter_mean[0]*image_size_x_y[1]*self.ym_per_pix + left_fit_meter_mean[1])**2)**1.5) / np.absolute(2*left_fit_meter_mean[0])
        right_curverad = ((1 + (2*right_fit_meter_mean[0]*image_size_x_y[1]*self.ym_per_pix + right_fit_meter_mean[1])**2)**1.5) / np.absolute(2*right_fit_meter_mean[0])

        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m
    
        return left_fit_mean, right_fit_mean, left_fit_meter_mean, right_fit_meter_mean, left_curverad, right_curverad
        
    def cal_car_offset(self, image_size_x_y, left_fit, right_fit): 
        lift_line_postion = left_fit[0]*image_size_x_y[1]**2 + left_fit[1]*image_size_x_y[0] + left_fit[2]
        right_line_postion = right_fit[0]*image_size_x_y[1]**2 + right_fit[1]*image_size_x_y[0] + right_fit[2]
        lane_center_postion = (lift_line_postion + right_line_postion) / 2
        camera_center = image_size_x_y[0]/2
        car_center = camera_center
        car_offset = (car_center - lane_center_postion) * self.xm_per_pix
        return car_offset 
        
        
    def argument_display(self, rgb_image, left_fit, right_fit, left_curverad, right_curverad, car_offset) :
        # Define conversions in x and y from pixels space to meters
        ploty = np.linspace(0, 700, num=700)# to cover same y-range as image  
        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Create an image to draw the lines on
        color_warp = np.zeros_like(rgb_image).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        #print()
        newwarp = cv2.warpPerspective(color_warp, self.ivs_perspect_matrix, (rgb_image.shape[1], rgb_image.shape[0])) 
        # Combine the result with the original image
        output_image = cv2.addWeighted(rgb_image, 1, newwarp, 0.3, 0)   
  
        text_font = cv2.FONT_HERSHEY_DUPLEX
        text_color = (255, 255, 255)
        curv_text = 'Curve radius left: {0:04.2f} m right: {1:04.2f} m'.format(left_curverad, right_curverad)
        cv2.putText(output_image, curv_text, (40, 70), text_font, 1, text_color, 2, cv2.LINE_AA)
        offset_text = 'Car offset: {:04.3f} m'.format(car_offset)
        cv2.putText(output_image, offset_text, (40, 120), text_font, 1, text_color, 2, cv2.LINE_AA)      
        return output_image

       
    def dectect_lane_line(self, rgb_image):
        image_size_x_y = (rgb_image.shape[1], rgb_image.shape[0])       
        undist_image = self.image_undistort(rgb_image)   
        combined_binary, color_binary = self.highlight_lane_line(undist_image)
        warped_image, color_warped_image = self.image_perspective(combined_binary)
        left_right_pixels = None                
        if (not self.working) or (self.need_reset) :
            left_right_pixels = self.hist_detect_lane(warped_image)
        else :
            left_right_pixels = self.hist_detect_lane_with_filter(warped_image, self.left_fit, self.right_fit)
            
        current_left_fix, current_right_fix = self.cal_parabola_pixels(image_size_x_y, left_right_pixels) 
        left_fit_mean, right_fit_mean, left_fit_cr_mean, right_fit_cr_mean, left_curverad, right_curverad = self.lane_line_postprocess(image_size_x_y, current_left_fix, current_right_fix)
        
        if not self.working :
            return rgb_image                        
        
        car_offset = self.cal_car_offset(image_size_x_y, left_fit_mean, right_fit_mean)
        output_image = self.argument_display(undist_image, left_fit_mean, right_fit_mean, left_curverad, right_curverad, car_offset)
    
        return output_image

        
    
################################################################################
################################################################################
 
                 
def process_image():
    
    images = glob.glob('test_images/test*.jpg')
    print(len(images))

    src = np.float32([[565, 460],
                      [715, 460],
                      [1150, 720],
                      [130, 720]])                    
    dst = np.float32([[440, 0],
                      [950,0],
                      [950,720],
                      [440, 720]])  
                      
    lane_line = C_LaneLine_t()
    lane_line.load_camera_parameter()
    lane_line.calc_m_mivs(src, dst)
    
    for image_file_path in images:
        # Read in an image
        left_right_list = []
        rgb_image = mpimg.imread(image_file_path) 
        image_size_x_y = (rgb_image.shape[1], rgb_image.shape[0])       
        undist_image = lane_line.image_undistort(rgb_image)
        mpimg.imsave('output_images/test_images/undist_'+image_file_path.split('/')[-1], undist_image)        
        combined_binary, color_binary = lane_line.highlight_lane_line(undist_image)
        mpimg.imsave('output_images/test_images/binary_'+image_file_path.split('/')[-1], color_binary)
        warped_image, color_warped_image = lane_line.image_perspective(combined_binary)
        mpimg.imsave('output_images/test_images/warped_'+image_file_path.split('/')[-1], color_warped_image)
        left_right = lane_line.identify_lane_line_convolution(warped_image)
        left_right_list.append(left_right)
        left_fit_mean, right_fit_mean, left_fit_cr_mean, right_fit_cr_mean, left_curverad, right_curverad = lane_line.cal_lane_line(image_size_x_y, left_right_list)                        
        car_offset = lane_line.cal_car_offset(image_size_x_y, left_fit_mean, right_fit_mean)
        output_image = lane_line.argument_display(undist_image, left_fit_mean, right_fit_mean, left_curverad, right_curverad, car_offset)
        mpimg.imsave('output_images/test_images/result_'+image_file_path.split('/')[-1], output_image)
  
                
def process_video():
    
       
    src = np.float32([[565, 460],
                      [715, 460],
                      [1150, 720],
                      [130, 720]])                    
    dst = np.float32([[440, 0],
                      [950,0],
                      [950,720],
                      [440, 720]]) 
             
    lane_line = C_LaneLine_t()
    lane_line.load_camera_parameter()
    lane_line.calc_m_mivs(src, dst)
    
    write_output = 'output_images/abc.mp4'

    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(lane_line.dectect_lane_line) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    


def run() :
    #process_image()
    process_video()
    
    
if __name__ == '__main__':
    run()   

  
