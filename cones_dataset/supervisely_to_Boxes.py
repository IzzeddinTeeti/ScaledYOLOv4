from mAP_for_yolo import Box
import json
import os
import uuid
import cv2

# Change that whrn appropriate
MAPPING = {
    'blue': 0,
    'orange': 1,
    'yellow': 2,
    'big orange': 1
}

def extractBoundingBoxes(json_path, min_width = 0, min_height = 0):
    fullList = []
    with open(json_path) as f:
        string = f.read()
        obj = json.loads(string)
        objects = obj["objects"]
        for numberOfCone in range(len(objects)):
            bbox = Box(str(uuid.uuid4()), 0, 0, 0, 0, 0)
            coords = objects[numberOfCone]['points']['exterior']
            bbox.w = coords[1][0] - coords[0][0]
            bbox.h = coords[1][1] - coords[0][1]
            bbox.x = coords[0][0] + int(bbox.w/2)
            bbox.y = coords[0][1] + int(bbox.h/2)
            bbox.cl = MAPPING[objects[numberOfCone]["classTitle"].split("_")[0]]
            if bbox.w > min_width and bbox.h > min_height:
                fullList.append(bbox)
    return fullList

def extractBB(img_path, txt_path):
    img = cv2.imread(img_path)
    img_shape = img.shape

    fullList = []
    with open(txt_path, "r") as detectionsFile:
        for line in detectionsFile.readlines():
            parts = list( map(lambda x: float(x), line.split()) )
            box = Box(str(uuid.uuid4()), parts[0], parts[1]*img_shape[1], parts[2]*img_shape[0], parts[3]*img_shape[1], parts[4]*img_shape[0])
            fullList.append(box)
    return fullList

def superviselyToDarknet(img_shape, sup_path, darknet_path, min_width=0, min_height=0):
    fullList = []
    with open(sup_path) as f:
        string = f.read()
        obj = json.loads(string)
        objects = obj["objects"]
        for numberOfCone in range(len(objects)):
            bbox = Box(str(uuid.uuid4()), 0, 0, 0, 0, 0)
            coords = objects[numberOfCone]['points']['exterior']
            bbox.w = coords[1][0] - coords[0][0]
            bbox.h = coords[1][1] - coords[0][1]
            bbox.x = coords[0][0] + int(bbox.w/2)
            bbox.y = coords[0][1] + int(bbox.h/2)
            bbox.cl = MAPPING[objects[numberOfCone]["classTitle"].split("_")[0]]
            if bbox.w > min_width and bbox.h > min_height:
                fullList.append(bbox)
    
    with open(darknet_path, "w") as detectionsFile:
        for box in fullList:
            string = "{} {} {} {} {}".format(box.cl, box.x/img_shape[1], box.y/img_shape[0], box.w/img_shape[1], box.h/img_shape[0])
            detectionsFile.write(string+"\n")