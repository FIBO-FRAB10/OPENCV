import numpy as np
import cv2
thres = 0.55 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280) #width
cap.set(4,480)  #height
cap.set(10,100)  #brightness

objectLabelFile = 'coco.names'
with open(objectLabelFile,'r') as f:
    objectNames = f.read().rstrip('\n').split('\n')

configFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsFile = 'frozen_inference_graph.pb'

model = cv2.dnnDetectionModel(weightsFile,configFile)
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

while cap.isOpened():
    ,frame = cap.read()
    frame=cv2.flip(frame,1) #flipHor
    classIds, confs, bbox = model.detect(frame,thres)
    if len(classIds) > 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            confsRate=f" {confs[0]*100:.0f}%"
            cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
            cv2.putText(frame,objectNames[classId-1]+confsRate,(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Output',frame)
    if cv2.waitKey(1) & 0xFF== ord('a') : break
cap.release()
cv2.destroyAllWindows()
