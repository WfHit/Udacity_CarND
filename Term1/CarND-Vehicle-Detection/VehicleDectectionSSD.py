#/usr/bin/env python3
'''
Vehicle Detection image scaled and channel order
'''
import pickle
import cv2
import numpy as np
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from scipy.misc import imread, imresize
from scipy.ndimage.measurements import label
#from scipy.ndimage import binary_closing, binary_opening
from moviepy.editor import VideoFileClip

from ssd.ssd_utils import BBoxUtility
from ssd.ssd import SSD300 as SSD

################################## Search Car ##################################
class DetectCars(object):
    """ Class for testing a trained SSD model and return boxes containing cars     
        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.                       
            model:       An SSD model. It should already be trained .                       
            model_input_shape: The shape that the model expects for its input, as a tuple, for example (300, 300, 3)                         
            image_shape: The shape of input image, as a tuple, for example (720, 1280, 3) 
    """
    def __init__(self, class_names, model, model_input_shape, image_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.bbox_util = BBoxUtility(self.num_classes)
        self.model_input_shape = model_input_shape
        self.image_width = image_shape[0]
        self.image_height = image_shape[1]
        self.hit_window_list = []
        
    def search_cars(self, rgb_image, conf_thresh = 0.6):
        """ search cars on rgb image and return boxes contain cars and confidence larger than conf_thresh.
        
        # Arguments
        rgb_image: Image on which, ssd search cars.
        conf_thresh: Threshold of confidence. Any boxes with lower confidence are not visualized.
        """
        # Resize input image to model image
        model_im_size = (self.model_input_shape[0], self.model_input_shape[1])    
        model_image = cv2.resize(rgb_image, model_im_size)
    
        # Use model to predict 
        inputs = [image.img_to_array(model_image)]
        tmp_inp = np.array(inputs)
        x = preprocess_input(tmp_inp)
        y = self.model.predict(x)
        
        # Get boxes result
        results = self.bbox_util.detection_out(y)
        
        boxes_list = []
        if len(results) > 0 and len(results[0]) > 0:
            # Interpret output, only one frame is used 
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            for i in range(top_conf.shape[0]):
                class_num = int(top_label_indices[i])
                if self.class_names[class_num] == 'car' :  
                    xmin = int(round(top_xmin[i] * self.image_width))
                    ymin = int(round(top_ymin[i] * self.image_height))
                    xmax = int(round(top_xmax[i] * self.image_width))
                    ymax = int(round(top_ymax[i] * self.image_height))                     
                    boxes_list.append([xmin, ymin, xmax, ymax])               
        return boxes_list
        
############################### Augmented Display ###############################
    def add_heat(self, orignal_image, hit_windows):
        '''
        Loop over hit_windows to create a heatmap
        '''
        heatmap = np.zeros_like(orignal_image[:,:,0]).astype(np.float)
        # Iterate through list of bboxes      
        for xmin, ymin, xmax, ymax in hit_windows:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[int(ymin):int(ymax), int(xmin):int(xmax)] += 1
        # Return updated heatmap
        return heatmap

    def filter_heatmap(self, heatmap, threshold):
        '''
        apply threshold to heatmap to get rid of the interference 
        '''
        filtered_heatmap = np.zeros_like(heatmap).astype(np.float)
        # Zero out pixels below the threshold
        filtered_heatmap[heatmap > threshold] = 255
        filtered_heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return filtered_heatmap

    def label_bboxes(self, filtered_heatmap):
        '''
        draw a box to indicate there is a car
        '''
        labels = label(filtered_heatmap)
        # Iterate through all detected cars
        bbox_list = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bbox_list.append(bbox)
        # Return the box
        return bbox_list
    
    def augmented_display(self, orignal_image, hit_windows) :
        '''
        augmented display pipeline
        '''
        # Add heat to each box in box list
        heatmap = self.add_heat(orignal_image, hit_windows)
        # Apply threshold to help remove false positives
        filtered_heatmap = self.filter_heatmap(heatmap, threshold=1)
        # Find final boxes from heatmap using label function
        # Indicate the cars on image 
        bbox_list = self.label_bboxes(filtered_heatmap)
        #print(bbox_list)
        draw_img = np.copy(orignal_image)

        for bbox in bbox_list:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (0, 255, 0), 3)
        return draw_img, heatmap, filtered_heatmap

    def image_prcess(self, rgb_image):
        '''
        process a single images
        '''
        hit_windows = self.search_cars(rgb_image)
        for xmin, ymin, xmax, ymax in hit_windows :
            if len(self.hit_window_list) < 10 :
                self.hit_window_list.append([xmin, ymin, xmax, ymax])
            else :
                temp_list = self.hit_window_list[1:]
                temp_list.append([xmin, ymin, xmax, ymax])
                self.hit_window_list = temp_list
        if len(self.hit_window_list) > 0 :
            rgb_image, heatmap, filtered_heatmap = self.augmented_display(rgb_image, self.hit_window_list)
            #cv2.rectangle(rgb_image,(xmin, ymin),(xmax, ymax),(0, 0, 255), 4)
        return rgb_image
        
##################################### Run #####################################        
def process_video():    
    '''
    Detect cars on video
    '''
    mdl_input_shape = (300,300,3)
    img_input_shape = (1280,720,3)
    
    # Classes for VOC
    class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
    NUM_CLASSES = len(class_names)
    model = SSD(mdl_input_shape, num_classes=NUM_CLASSES)
    model.load_weights('./ssd/weights_SSD300.hdf5') 
    #class_names, model, model_input_shape, image_shape
    cars_detector = DetectCars(class_names, model, mdl_input_shape, img_input_shape)
    #read in video
    write_output = 'output_images/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(cars_detector.image_prcess) #NOTE: this function expects color images!!
    white_clip.write_videofile(write_output, audio=False)
    
    
if __name__ == '__main__':
    process_video()
