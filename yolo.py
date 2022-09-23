import numpy as np
import argparse
import time 
import cv2
import os
#Checking for GPU/CUDA support
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
#Arguments for the input image and the yolo-weights folder are recieved here
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path for the input image")
ap.add_argument("-y", "--yolo", required=True, help="Path for the Yolo Weights directory")
ap.add_argument("-c", "--confidence", type=float, default=0.25, help="Minimum Probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
labelsPath = os.path.sep.join([args["yolo"], "vehicle_coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#Colors for different labels for detected vehicles in image/video
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([args["yolo"], "yolov3-tiny-custom_final.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3-tiny-custom.cfg"])

#Load weights into OpenCV
print("[INFO] Loading YOLOV3-Tiny weights from the file......")
print("[INFO] Config Path: ",configPath)
print("[INFO] Weights Path: ", weightsPath)

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#grab input image from OpenCV here
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

#Determining the output layer labels from the trained darknet model
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#constructing a blob for the imput image sent and then perform a forward pass
#of the YOLO object detector, this outputs bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

#Shows time duration taken for detections in the image
print("[INFO] YOLO took {:.6f} seconds".format(end - start))
#Initialise bounding boxes arrays, confidence scores and classIDs
boxes = []
confidences = []
classIDs = []
#loop to iterate over all the layers in the yolo model provided
for output in layerOutputs:
    #loop over each iteration in the neural net
    for detection in output:
        #Extract classIDs and confidence of current object detected
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        #filtering out weak predictions by setting the threshold
        #from the arguments passed in the command
        if confidence > args["confidence"]:
            #scale bounding box coordinates relative to the size
            #of the image, keeping in mind that YOLO returns the center
            #coordinates of the bounding box followed by width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            #use center to get top left co-ordinates of bounding box
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))
            #update list of bounding boxes and their corresponding classes
            #to the array of bounding boxes and classIDs
            boxes.append([x,y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
if len(idxs) > 0:
    #loop over all the indexes detected
    for i in idxs.flatten():
        #get bounding box co-ordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        #draw bounding  box rectangles and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x,y),(x+w, y+h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
cv2.imshow("Image", image)
cv2.waitKey(0)