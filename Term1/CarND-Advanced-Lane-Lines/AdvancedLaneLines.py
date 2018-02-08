#/usr/bin/env python3
'''
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
    image = cv2.imread(image_file_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Record image shape
    image_shape = gray.shape[::-1]
    
    # Find the chessboard corners
    ret, img_corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, draw corners
    if ret == True:
        # Add corners in image points
        imgpoints.append(img_corners)
        # Add corners in object points
        objpoints.append(objp_corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(image, (nx, ny), img_corners, ret)
        plt.imshow(image)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

# Undistort image
'''
f, ax = plt.subplots(len(images), 2, figsize=(24, 9))
f.tight_layout()
ax_counter = 0
for image_file_path in images:
    # Read in an image
    image = cv2.imread(image_file_path)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    
    ax[ax_counter+1].imshow(image.squeeze())
    ax[ax_counter+1].set_title('Original Image', fontsize=50)
    ax[ax_counter+2].imshow(undistorted.squeeze())
    ax[ax_counter+2].set_title('Undistorted Image', fontsize=50)
    ax_counter += 2
   
fig = plt.figure(figsize=(150, 150))
counter = 0
for image_file_path in images:
    # Read in an image
    image = cv2.imread(image_file_path)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    image = image.squeeze()
    undist = undist.squeeze()
    img = fig.add_subplot(len(images), 2, counter+1)
    img.imshow(image)
    img.set_title('Original Image')
    img = fig.add_subplot(len(images), 2, counter+2)
    img.imshow(undist)
    img.set_title('Undistorted Image')
    plt.show()

plt.show()
'''
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

for image_file_path in images:
    # Read in an image
    image = cv2.imread(image_file_path)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    image = image.squeeze()
    undist = undist.squeeze()
    plt.imshow(image)
    #plt.set_title('Original Image')
    plt.imshow(undist)
    #plt.set_title('Undistorted Image')
    plt.show()




