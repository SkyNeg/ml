# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import ntpath
import time

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_build_path = 'G:/My documents/Work/Krossover/openpose/build'
openpose_model_folder = 'G:/My documents/Work/Krossover/openpose/models/'
try:
    # Windows Import
    # Change these variables to point to the correct folder (Release/x64 etc.) 
    sys.path.append(openpose_build_path + '/python/openpose/Release');
    os.environ['PATH']  = os.environ['PATH'] + ';' + openpose_build_path + '/x64/Release;' +  openpose_build_path + '/bin;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

parser = argparse.ArgumentParser()
parser.add_argument("-i", default="", help="Input video file")
parser.add_argument('-o', type=str, default='out', help='Out folder')
parser.add_argument('-f', type=int, default=1, help='Process every f frame')
parser.add_argument('-mode', type=int, default=0, help='Output type: 1 - csv, 2 - video, 4 - tbd')
args = parser.parse_args()

FRACTION = args.f

filename = ntpath.basename(args.i)
out_videofilename = args.o + '/' + os.path.splitext(filename)[0] + '_openpose_out' + '.avi'
out_csvfilename = args.o + '/' + os.path.splitext(filename)[0] + '_openpose_out' + '.csv'

cap = cv2.VideoCapture(args.i)
oWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
oHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
oFps = int(cap.get(cv2.CAP_PROP_FPS))

if args.mode & 1 != 0:
    fwriter = open(out_csvfilename, 'w')
    fwriter.write('Frame,Person,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25\n')

if args.mode & 2 != 0:
    size = oWidth, oHeight
    vwriter = cv2.VideoWriter(out_videofilename, cv2.VideoWriter_fourcc(*'MJPG'), oFps / FRACTION, (oWidth, oHeight))

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = openpose_model_folder

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    start = time.time()
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1

        if (counter % FRACTION != 0):
            continue

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        if args.mode & 1 != 0:
            personCounter = 0
            # Write data to csv file
            for poseKeypoint in datum.poseKeypoints:
                personCounter += 1
                fwriter.write(str(counter) + ',' + str(personCounter) +  ',')# + obj.label + ',' + str(obj.xmin) + ',' + str(obj.ymin) + ',' + str(obj.xmax) + ',' + str(obj.ymax) + ',' + str(obj.threshold) + '\n')
                for point in poseKeypoint:
                    fwriter.write(str(point) + ',')
                fwriter.write('\n')

        # Write frame to output video
        if args.mode & 2 != 0:
            new_image = datum.cvOutputData
            vwriter.write(new_image)

except Exception as e:
    print(e)
    sys.exit(-1)

cap.release()