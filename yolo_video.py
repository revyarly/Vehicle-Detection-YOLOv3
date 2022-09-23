import numpy as np
import argparse
import imutils
import time
import cv2
import os
#Checking for GPU/CUDA support
count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)
#Arguments for the input image and the yolo-weights folder are recieved here
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input", required=True, help="Path to input video")
ap.add_argument("-o","--output", required=True, help="Path to output video")
ap.add_argument("-y", "--yolo", required=True, help="Path to YOLO models and weights")
ap.add_argument("-c", "--confidence", type=float, default=0.25, help="Minimum probability detection rate")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold for applying non-maxima suppression")
args = vars(ap.parse_args())
labelsPath = os.path.sep.join([args["yolo"],"vehicle_coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
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
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#initialise the video stream, pointer to output video file, and fromm dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
#Try to count the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
#If an error occurs then print the appropriate solution
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
#loop over frames from the videoo file stream
while True:
    #read the next frame from the file
    (grabbed, frame) = vs.read()
    #if frame is not grabbed we have reached end of stream
    if not grabbed:
        break
    #if frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    #contruct blob from each frame and perform a forward pass of the YOLO object detetctor
    #giving us bounding boxes and associated classIDs
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    #initialise our lists of detected bounding boxes, confidences, and classIDs
    boxes = []
    confidences = []
    classIDs = []
    #loop over each layer of the neural neet outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            #extract the classID and confidence score of the current
            #object detected
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
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                #update list of bounding boxes and their corresponding classes
                #to the array of bounding boxes and classIDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #ensure atleat one detection exists
        if len(idxs) > 0:
            #loop over the indexes we are keeping
            for i in idxs.flatten():
                #extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if writer is None:
        #initialise our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
    writer.write(frame)
print("[INFO] cleaning up...")
writer.release()
vs.release()