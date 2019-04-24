###########
# Imports #
###########
from __future__ import division
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import argparse
import ntpath

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import cv2

#############
# Env setup #
#############
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

#############
# Variables #
#############
parser = argparse.ArgumentParser(description="RFCN tester")
parser.add_argument('-i', type=str, help='Input file')
parser.add_argument('-o', type=str, help='Out folder')
parser.add_argument('-m', type=str, default="", help='Model')
parser.add_argument('-mode', type=int, default=0, help='Output type: 1 - csv, 2 - video, 4 - tbd')
parser.add_argument('-f', type=int, default=1, help='Process every f frame')
opt = parser.parse_args()

# Path to folder with frozen detection graph.
modeldir = 'G:/My documents/Work/Krossover/Coco/'
trainedmodel = opt.m

PATH_TO_CKPT = modeldir + trainedmodel + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
NUM_CLASSES = 80
FRACTION = opt.f
filename = ntpath.basename(opt.i)
out_videofilename = opt.o + '/' + os.path.splitext(filename)[0] + '_' + trainedmodel + '_out' + '.avi'
out_csvfilename = opt.o + '/' + os.path.splitext(filename)[0] + '_' + trainedmodel + '_out' + '.csv'

#################################################
# Load a (frozen) Tensorflow model into memory.  ## Load a (frozen) Tensorflow model into memory. #
#################################################
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

cap = cv2.VideoCapture(opt.i)
oWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
oHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
oFps = int(cap.get(cv2.CAP_PROP_FPS))

if opt.mode & 1 != 0:
    fwriter = open(out_csvfilename, 'w')
    fwriter.write('Frame,label,xmin,ymin,xmax,ymax,threshold\n')

if opt.mode & 2 != 0:
    size = oWidth, oHeight
    vwriter = cv2.VideoWriter(out_videofilename, cv2.VideoWriter_fourcc(*'MJPG'), oFps / FRACTION, (oWidth, oHeight)) 

#####################
# Loading label map #
#####################
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

###############
# Helper code #
###############
class DetectedObject(object):
    label = ""
    threshold = 0.0
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0

    def __init__(self, label, xmin = 0, ymin = 0, xmax = 0, ymax = 0, threshold = 0.0):
        self.label = label
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.threshold = threshold

def get_objects(output_dic, h, w):
    objects = []
    counter = 0
    for n in range(len(output_dic['detection_scores'])):
        threshold = output_dic['detection_scores'][n]
        if output_dict['detection_scores'][n] > 0.70:
            # Calculate position
            ymin = int(output_dic['detection_boxes'][n][0] * h)
            xmin = int(output_dic['detection_boxes'][n][1] * w)
            ymax = int(output_dic['detection_boxes'][n][2] * h)
            xmax = int(output_dic['detection_boxes'][n][3] * w)

            # Find label corresponding to that class
            for cat in categories:
                if cat['id'] == output_dic['detection_classes'][n]:
                    label = cat['name']

                    ## extract every person
                    if label == 'person' or label == 'sports ball':
                        ## if box is not too big:
                        area_box = (ymax - ymin) * (xmax - xmin)
                        area_img = h * w
                        boxPer = area_box/area_img
                        if boxPer <= 0.15:
                            obj = DetectedObject(label, xmin, ymin, xmax, ymax, threshold)
                            objects.append(obj)
    return objects

def get_frame_from_image(img, objects, index = 0):
    w, h, _ = img.shape

    # Turned off display to avoid information overload
    # vis_util.visualize_boxes_and_labels_on_image_array(image_np, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], category_index, instance_masks=output_dict.get('detection_masks'), use_normalized_coordinates=True, min_score_thresh=0.60, line_thickness=2)

    frame_number = index

    ## print color next to the person
    for obj in objects:
        cv2.rectangle(img, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), (0, 255, 0), 1)
        cv2.putText(img, obj.label, (obj.xmin, obj.ymin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)

    #cv2.destroyAllWindows()
    return img

# Running the tensorflow session
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        counter = 0
        fac = 3

        while True:
            ret, image_np = cap.read()
            if not ret:
                break

            counter += 1

            if (counter % FRACTION != 0):
                continue

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image_np, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            objects = get_objects(output_dict, oHeight, oWidth)
            # Write data to csv file
            if opt.mode & 1 != 0:
                for obj in objects:
                    fwriter.write(str(counter) +  ',' + obj.label + ',' + str(obj.xmin) + ',' + str(obj.ymin) + ',' + str(obj.xmax) + ',' + str(obj.ymax) + ',' + str(obj.threshold) + '\n')

            # Write frame to output video
            if opt.mode & 2 != 0:
                new_image = get_frame_from_image(image_np, objects, counter)
                vwriter.write(new_image)

        cap.release()