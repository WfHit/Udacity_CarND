from styx_msgs.msg import TrafficLight
import os , sys
import numpy as np
import tensorflow as tf
import time
import threading

class TLClassifier(object):
    def __init__(self):
    
        self.cnn_running = False      
        self.start_event = threading.Event()
        self.tl_class = -1
        self.image = None 
        self.tl_classity_run = True   
        self.missing_count = 0
        
        thread_cnn = threading.Thread(target=self.task_tl_classity, name='task_tl_classity')
        thread_cnn.start()
        #TODO load classifier    
        #pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction
        state = TrafficLight.UNKNOWN
        
        if not self.start_event.isSet():  
            self.image = image
            #self.missing_count = 0           
            if self.tl_class == 1 :
                state = TrafficLight.GREEN
            elif self.tl_class == 2 :
                state = TrafficLight.RED
            elif self.tl_class == 3 :
                state = TrafficLight.YELLOW
            elif self.tl_class == 4 :
                state = TrafficLight.UNKNOWN  
            else :
                state = TrafficLight.UNKNOWN 
            self.start_event.set()    
        else :
            self.missing_count += 1        
        return state

             
    def task_tl_classity(self):
    
        #print "os.getcwd()=%s" % os.getcwd()    
        self.model_path = 'light_classification/tl_classifier/models/ssd_udacity/frozen_inference_graph.pb'
        self.class_number = 4

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:     
            print(' class_name ', ' missing_count ' , ' time_cost ', ' class_scores ')       
            while self.tl_classity_run:
              #print('thread %s is running...' % threading.current_thread().name)
              self.start_event.wait()
              t_start = time.clock()
              # Expand dimension since the model expects image to have shape [1, None, None, 3].
              img_expanded = np.expand_dims(self.image, axis=0)
              (boxes, scores, classes, num) = sess.run([self.d_boxes, self.d_scores, self.d_classes, self.num_d], feed_dict={self.image_tensor: img_expanded})
              boxes = np.squeeze(boxes)
              scores = np.squeeze(scores)
              classes = np.squeeze(classes).astype(np.int32)
              if scores[0] > 0.6:
                  self.tl_class = classes[0]  
              self.start_event.clear()
              t_cost = time.clock() - t_start
              if self.tl_class == 1 :
                  class_name = 'GREEN  '
              elif self.tl_class == 2 :
                  class_name = 'RED    '
              elif self.tl_class == 3 :
                  class_name = 'YELLOW '
              else  :
                  class_name = 'UNKNOWN' 
              print(class_name, self.missing_count, t_cost, round(scores[0],3))
