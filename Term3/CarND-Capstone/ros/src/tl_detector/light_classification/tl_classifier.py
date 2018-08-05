from styx_msgs.msg import TrafficLight
import os , sys
import numpy as np
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier    
        
        #self.model_path = 'src/tl_detector/light_classification/tl_classifier/models/ssd_sim/frozen_inference_graph.pb'
        self.label_path = 'light_classification/tl_classifier/data/udacity_label_map.pbtxt'
        self.class_number = 4
          
        self.label_map = label_map_util.load_labelmap(self.label_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.class_number, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        
        print "os.getcwd()=%s" % os.getcwd()
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('light_classification/tl_classifier/models/ssd_sim/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options) )
        #pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            #image_np = self.load_image_into_numpy_array(image)
            img_expanded = np.expand_dims(image, axis=0)
            #img_expanded = image #np.expand_dims(image_np, axis=0)  
            (boxes, scores, classes, num) = self.sess.run([self.d_boxes, self.d_scores, self.d_classes, self.num_d], feed_dict={self.image_tensor: img_expanded})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            class_name = self.category_index[classes[0]]['name']
            print('{}'.format(class_name), scores[0])
            if scores[0] > 0.6:
              state = TrafficLight.UNKNOWN
              if classes[0] == 1 :
                  state = TrafficLight.GREEN
              elif classes[0] == 2 :
                  state = TrafficLight.RED
              elif classes[0] == 3 :
                  state = TrafficLight.YELLOW
              elif classes[0] == 4 :
                  state = TrafficLight.UNKNOWN  
              else :
                  state = TrafficLight.UNKNOWN                
              return state

