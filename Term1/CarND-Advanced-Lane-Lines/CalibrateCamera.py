#/usr/bin/env python3
'''
Calibrate Camera
'''
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Read in all the cal images
images = glob.glob('camera_cal/calibration*.jpg')
print(len(images))

# List for object points
objpoints = []
# List for image points
imgpoints = []

# The number of inside corners in x
nx = 9
# The number of inside corners in y
ny = 6

# Object pionts of the images are same
objp_corners = np.zeros((nx*ny, 3), np.float32)
objp_corners[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Image shape
image_shape = None
        
for image_file_path in images:
    # Read in an image
    bgr_image = cv2.imread(image_file_path.strip()) 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    # Record image shape
    image_shape = gray.shape[::-1]
    # Find the chessboard corners
    ret, img_corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, add to imgpoints list
    if ret == True:
        # Add corners in image points
        imgpoints.append(img_corners)
        # Add corners in object points
        objpoints.append(objp_corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

cal_pickle = {}
cal_pickle["mtx"] = mtx
cal_pickle["dist"] = dist

with open('calibrate_camera.pkl', 'wb') as f:
    pickle.dump(cal_pickle, f)
    
'''
with open('calibrateCamera.pkl', 'rb') as f:
    cal_pickle = pickle.load(f)
'''

# Undistort image
for image_file_path in images:
    fig = plt.figure(figsize=(50, 50))
    #fig.tight_layout()
    # Read in an image
    bgr_image = cv2.imread(image_file_path.strip()) 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(rgb_image, mtx, dist, None, mtx)
    rgb_image = rgb_image.squeeze()
    undist = undist.squeeze()
    mpimg.imsave('output_images/camera_cal/'+image_file_path.split('/')[-1], undist)
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(rgb_image)
    ax.set_title('Original Image', fontsize=10)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(undist)
    ax.set_title('Undistorted Image', fontsize=10)
    plt.savefig('output_images/'+image_file_path)
    plt.show()

