###########
# Imports #
###########
import os
import argparse
from imageai.Detection import VideoObjectDetection
from io import StringIO
import ntpath

#############
# Variables #
#############
parser = argparse.ArgumentParser()
parser.add_argument('-i', type = str, help = 'Input file')
parser.add_argument('-o', type = str, help = 'Out folder')
parser.add_argument('-m', type = str, default = "", help = 'Model')
parser.add_argument('-f', type = int, default = 1, help = 'Process every f frame')
opt = parser.parse_args()

FRACTION = opt.f

filename = os.path.splitext(ntpath.basename(opt.i))[0]
model = os.path.splitext(ntpath.basename(opt.m))[0]
out_videofilename = opt.o + '/' + filename + '_' + model + '_out'
out_csvfilename = opt.o + '/' + filename + '_' + model + '_out' + '.csv'

execution_path = os.getcwd()
###############
# Helper code #
###############
def frameFunc(frame_number, output_array, output_count):
    for obj in output_array:
        fwriter.write(str(frame_number) +  ',' + obj['name'] + ',' + str(obj['box_points'][0]) + ',' + str(obj['box_points'][1]) + ',' + str(obj['box_points'][2]) + ',' + str(obj['box_points'][3]) + ',' + str(obj['percentage_probability']) + '\n')

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(opt.m)
detector.loadModel()

fwriter = open(out_csvfilename, 'w')
fwriter.write('Frame,label,xmin,ymin,xmax,ymax,threshold\n')

video_path = detector.detectObjectsFromVideo(input_file_path = opt.i,
                                             output_file_path = out_videofilename,
                                             frame_detection_interval = FRACTION,
                                             #frames_per_second = 30,
                                             per_frame_function = frameFunc,
                                             minimum_percentage_probability = 70)
print(video_path)