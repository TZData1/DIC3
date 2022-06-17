# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from PIL import ImageDraw
import os
#import detect
#import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
from tensorflow.python.keras.utils.data_utils import get_file

#tenserflow was added as extra library
import tensorflow as tf
import time


#used for generating colors for classes
np.random.seed(20)
cacheDir = "./pretrained_models"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

app = Flask(__name__)

def createBoundingBox(image,model,modelName):
    start = time.time()
    classesList,colorList = readClasses("coco.names")
    #converting to RGB format
    #np array
    inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
    #converting to tensor
    inputTensor = tf.convert_to_tensor(inputTensor,dtype = tf.uint8)

    #We are expanding dimensions because we are working with batches
    inputTensor = inputTensor[tf.newaxis,...]

    #detections are dictionaries and we are extracting each feature
    detections = model(inputTensor)
    bboxs = detections['detection_boxes'][0].numpy()
    classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
    classesScores = detections['detection_scores'][0].numpy()

    #we need this for calculation of location for bounding boxes
    imH,imW, imC = image.shape
    #In the first implementation we get a lot of overlaping bounding boxes
    #here we are declaring that we can have maximum of 50 bounding boxes, 50% of overlap is acceptable and confidence is also 50%.
    bboxIdx = tf.image.non_max_suppression(bboxs,classesScores,max_output_size= 50, iou_threshold=0.5,score_threshold=0.5)

    bounding_box_coords = []
    if len(bboxIdx) != 0:
        for i in bboxIdx:
            bbox = tuple(bboxs[i].tolist())
            #Confidence score for the particular object
            classConfidence = round(100*classesScores[i])
            classIndex = classIndexes[i]
            #extracting label for the object
            classLabelText = classesList[classIndex]
            #extracting color of boundingbox for the class
            classColor = colorList[classIndex]
            displayText = '{}: {}%'.format(classLabelText,classConfidence)

            #unpack bounding box so we can get values of the pixels on x and y axis
            #they are relative to widht and the height of the image and they are not absolute locations


            ymin, xmin, ymax, xmax = bbox
            xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
            tuple4 = (xmin,xmax,ymin,ymax)
            bounding_box_coords.append(tuple4)
    end = time.time()
    time_passed = end - start
    #Writing the time for processing for each image into a file, with that we will calculate average time for processing in total.
    with open("time_for_processing", "a") as f:
        f.write(str(time_passed) + "\n")
    return bounding_box_coords


def predictImage(imagePath,model,modelName):
    #image = imagePath if we are sending image to this function, if we are sending path to image then we need cv2.imread function
    image = cv2.imread(imagePath)
    coords = createBoundingBox(image,model,modelName)
    return coords


def loadModel(modelURL):
    modelName = downloadModel(modelURL)
    print("Loading model " + modelName)
    model = tf.saved_model.load(os.path.join(cacheDir,"checkpoints",modelName,"saved_model"))
    print("model " + modelName + "loaded")
    return model,modelName

def downloadModel(modelURL):
    #downloading model from specific url
    #if model is already downloaded we will not download it again
    fileName = os.path.basename(modelURL)
    modelName = fileName[:fileName.index('.')]
    os.makedirs(cacheDir,exist_ok=True)

    #extracting model into "checkpoints" subdirectory
    get_file(fname = fileName, origin = modelURL,cache_dir = cacheDir, cache_subdir = "checkpoints", extract = True)
    return modelName


def readClasses(classesFilePath):
    #reading classes from coco.names file we have 92 classes in total (so we will have 92 colors eg one for each class)
    with open(classesFilePath,'r') as f:
        classesList = f.read().splitlines()
    colorList = np.random.uniform(low = 0, high = 255, size = (len(classesList),3))
    return classesList,colorList


def detection_loop(image,model,modelName):
   coords = predictImage(image,model,modelName)
   return coords



#initializing the flask app
app = Flask(__name__)

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  #img = request.files["image"].read()
  #image = Image.open(io.BytesIO(img))
  #data_input = request.args['input']
  data_input = request.values.get('input')
  output = request.values.get('output')
  #output = request.form.get('output')

  path = data_input
  filename_image = {}
  
  input_format = ["jpg", "png", "jpeg"]
  if data_input.find(".") != -1:
      print(data_input + " is a file")
      split_data_input = data_input.split(".", 1)
      if data_input.endswith(tuple(input_format)):
          print("INPUT FORMAT: %s IS VALID" % split_data_input[1])
          path_splitted = []
          path_splitted = re.split('/', data_input)
          filename = path_splitted[len(path_splitted)-1]
          filename_image[filename] = Image.open(data_input)
          path = os.path.dirname(data_input)+"/"
  else:
      print(data_input + " is a path with the following files: ")
      for filename in os.listdir(data_input):
          image_path = data_input + filename
          filename_image[filename] = Image.open(image_path)
          print("  " + filename)
  
  detection_loop(filename_image, path, output)
  
  status_code = Response(status = 200)
  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    #loadModel(modelURL)
    model,modelName = loadModel(modelURL)
    print(detection_loop("000000146906.jpg",model,modelName))
    #app.run(debug = True, host = '0.0.0.0')
