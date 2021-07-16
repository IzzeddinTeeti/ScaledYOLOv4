import math
import numpy as np

'''
TODO: Describe the class
what is self.detected?
Can we have classes that can be used for both Perceptoin and Localization & Mapping?
In README of the GitLab project, include information about these classes and the coordinate frames
'''
class Box:
    def __init__(self, id, cl, center_x, center_y, width, height):
        self.id = id
        self.cl = int(cl)
        self.x = int(center_x)
        self.y = int(center_y)
        self.w = int(width)
        self.h = int(height)
        self.dist = None
        self.detected = False
        self.tags = list()
    
    # Distance in meters (rounded up)
    # v = (x,z) vector
    def setDistance(self, v):
        d = math.sqrt(v[0]**2 + v[1]**2)
        self.dist = math.ceil(d/1000)
    
    def __str__(self):
        return "Box {} ({},{})".format(self.cl, self.x, self.y)

def calcdist(pt1: Box, pt2: Box):
    return math.sqrt( (pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 )

# https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def getIoU(box1: Box, box2: Box):
    left1 = box1.x - box1.w/2
    right1 = box1.x + box1.w/2
    top1 = box1.y - box1.h/2
    bottom1 = box1.y + box1.h/2

    left2 = box2.x - box2.w/2
    right2 = box2.x + box2.w/2
    top2 = box2.y - box2.h/2
    bottom2 = box2.y + box2.h/2

    horizontal = sorted([left1, right1, left2, right2])
    vertical = sorted([top1, bottom1, top2, bottom2])

    if (horizontal == [left1, right1, left2, right2] or vertical == [top1, bottom1, top2, bottom2] or
        horizontal == [left2, right2, left1, right1] or vertical == [top2, bottom2, top1, bottom1]):
        IoU = 0
    else:
        intersection = (horizontal[2] - horizontal[1]) * (vertical[2] - vertical[1])
        area1 = box1.w * box1.h
        area2 = box2.w * box2.h
        union = area1 + area2 - intersection
        IoU = intersection/union
    
    return IoU

def printDebug(content, debugging):
    if debugging:
        print(content)

'''
Returns Precision, Recall and Average Precision for an image
detections = list of Box objects (detected)
gt = list of Box objects (ground truth)
'''
def analyseFrame(detections, gt, debugging=False):
    # Special cases. The can seriously mess the average 
    if len(detections) == 0:
        if len(gt) == 0:
            return 1, 1, 1
        else:
            return 1, 0, 0

    gt_correlations = {} #{"id": {"gt": Box, "dets": [list of matched detection boxes]}}

    # Pre-populate the gt_correlations object with ground truth boxes
    for count, gt_box in enumerate(gt):
        gt_correlations[gt_box.id] = {"gt": gt_box, "dets": []}

    # Mapping detections to ground truth (one detection can be mapped to maximum 1 ground truth - the closest GT is picked)
    # Basically, choose closest ground truth box for each detection box
    for detpoint in detections: # We need to allocate every detection bounding box to a ground truth box
        closest = None
        mindist = 2000 #some arbitrary large value
        for gtpoint in gt:
            dist_to_gt = calcdist(gtpoint, detpoint)
            # print(dist)
            if dist_to_gt < mindist:
                mindist = dist_to_gt
                closest = gtpoint
        gt_correlations[closest.id]["dets"].append(detpoint)
        #print("Paired " + str(detpoint) + " with " + str(closest))
    
    # Print and make sure the values of ground truths and the values of detections stayed the same
    printDebug("////", debugging)
    printDebug("len(gt): " + str(len(gt)), debugging)
    printDebug("len(detections): " + str(len(detections)), debugging)
    printDebug("len(gt_correlations): " + str(len(gt_correlations)), debugging)
    printDebug("# of all correlated detections: "
        + str(sum([len(g["dets"]) for g in gt_correlations.values()])), debugging)
    printDebug("////", debugging)

    R = 0.0
    P = 1.0
    true_positives = []     # put 1 if tp, 0 otherwise (fp)
    false_positives = []    # put 1 if fp, 0 otherwise (tp)
    totalGT = len(gt)
    totalDET = len(detections)

    # Parameter that defines whether 2 boxes (ground truth and detection) have significant "intersection over union" to be considered correct detection
    IoU_threshold = 0.5

    # The analyser. Calculates Recall, Precision and fills in the table for distance-detection plot
    for gt_id, obj in gt_correlations.items():
        printDebug("Ground truth: " + str(obj["gt"]), debugging)
        if len(obj["dets"]) > 0:
            bestIoU = 0
            best = None
            for det in obj["dets"]:
                #print("\t" + str(det), end=" ")
                IoU = getIoU(det, obj["gt"])
                if IoU > IoU_threshold:
                    if det.cl != obj["gt"].cl:
                        P -= 1/totalDET
                        false_positives.append(1)
                        true_positives.append(0)
                        printDebug("Misclassified", debugging)
                        printDebug("New P: " + str(P), debugging)
                    else:
                        if best is None:
                            R += 1/totalGT  # dont count multiple detections of the same object
                            obj["gt"].detected = True
                            false_positives.append(0)
                            true_positives.append(1)
                            printDebug("correct!", debugging)
                            printDebug("New R: " + str(R), debugging)
                        else:
                            P -= 1/totalDET
                            false_positives.append(1)
                            true_positives.append(0)
                            printDebug("already detected", debugging)
                            printDebug("New P: " + str(P), debugging)
                        if IoU > bestIoU:
                            best = det
                            bestIoU = IoU
                else:
                    P -= 1/totalDET
                    false_positives.append(1)
                    true_positives.append(0)
                    printDebug("IoU too low: " + str(IoU), debugging)
                    printDebug("New P: " + str(P), debugging)
        # else:
        #     print("\tNo detections")

    # print("P:{:.2%}, R:{:.2%}".format(P, R))
    # print("fp:{}".format(false_positives))
    # print("tp:{}".format(true_positives))

    # get a cumulative list
    fp_c = np.array(false_positives).cumsum()
    tp_c = np.array(true_positives).cumsum()

    # print("fpc:{}".format(fp_c))
    # print("tpc:{}".format(tp_c))

    # get recall and precision curves
    precision_curve =  tp_c / (tp_c + fp_c)
    recall_curve = tp_c / (totalGT + 1e-16)

    # print("Precision curve: {}".format(precision_curve))
    # print("Recall curve: {}".format(recall_curve))

    ap = compute_ap(recall_curve, precision_curve)
    # print("AP: {}".format(ap))
    return P, R, ap