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
        #plt.imshow(image)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

cal_pickle = {}
cal_pickle["mtx"] = mtx
cal_pickle["dist"] = dist

with open('calibrateCamera.pkl', 'wb') as f:
    pickle.dump(cal_pickle, f)
    
'''
with open('calibrateCamera.pkl', 'rb') as f:
    cal_pickle = pickle.load(f)
'''

# Undistort image

fig = plt.figure(figsize=(50, 50))
#fig.tight_layout()
counter = 0
for image_file_path in images:
    # Read in an image
    image = cv2.imread(image_file_path)
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    image = image.squeeze()
    undist = undist.squeeze()
    cv2.imwrite('output_images/camera_cal/'+image_file_path.split('/')[-1], undist)
    ax = fig.add_subplot(2, len(images), counter+1)
    ax.imshow(image)
    ax.set_title('Original Image', fontsize=10)
    ax = fig.add_subplot(2, len(images), len(images)+counter+1)
    ax.imshow(undist)
    ax.set_title('Undistorted Image', fontsize=10)
    counter += 1
plt.savefig('output_images/camera_cal/cal_camera.png')
plt.show()

