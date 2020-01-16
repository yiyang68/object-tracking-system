import imutils
import time
import cv2
import os
import numpy as np
from imutils.video import VideoStream

def yolo_detection(frame, W, H, net, ln, args):
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences, classIDs

def yolo_multitracker(args):
    # load the COCO class labels our YOLO model was trained on
    # labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    # LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    # np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    #                            dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    # vs = cv2.VideoCapture(args["video"])
    # writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    # try:
    #     prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
    #         else cv2.CAP_PROP_FRAME_COUNT
    #     total = int(vs.get(prop))
    #     print("[INFO] {} total frames in video".format(total))
    #
    # # an error occurred while trying to determine the total
    # # number of frames in the video file
    # except:
    #     print("[INFO] could not determine # of frames in video")
    #     print("[INFO] no approx. completion time can be provided")
    #     total = -1

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    # if a video path was not supplied, grab the reference to the web cam
    if not args.get("video", False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

    # otherwise, grab a reference to the video file
    else:
        vs = cv2.VideoCapture(args["video"])

    # loop over frames from the video stream
    i = 0
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame (so we can process it faster)
        frame = imutils.resize(frame, width=600)

        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = trackers.update(frame)

        # loop over the bounding boxes and draw then on the frame
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if i == 0:
            new_trackers = cv2.MultiTracker_create()
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            (boxes, confidences, classIDs)=yolo_detection(frame, W, H, net, ln, args)
            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                    args["threshold"])
            if len(idxs) > 0:
                for j in idxs.flatten():
                    (x, y) = (boxes[j][0], boxes[j][1])
                    (w, h) = (boxes[j][2], boxes[j][3])
                    box = (x, y, w, h)
                    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                    new_trackers.add(tracker, frame, box)
                trackers = new_trackers

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
        i += 1
        if i == 60:
            i = 0

    # if we are using a webcam, release the pointer
    if not args.get("video", False):
        vs.stop()

    # otherwise, release the file pointer
    else:
        vs.release()

    # close all windows
    cv2.destroyAllWindows()