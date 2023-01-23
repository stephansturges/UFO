#!/usr/bin/env python3
"""
The code is edited from docs (https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/)
We add parsing from JSON files that contain configuration
"""

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
import json
import blobconverter
import os
from time import monotonic

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='./model/best_openvino_2021.4_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='./model/best.json', type=str)
args = parser.parse_args()

mediaPath = "pics/"

nn_shape = 640

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))



# Create pipeline
pipeline = dai.Pipeline()
# Define sources and outputs

#NODES
nn = pipeline.create(dai.node.YoloDetectionNetwork)
xinFrame = pipeline.create(dai.node.XLinkIn)
nnOut = pipeline.create(dai.node.XLinkOut)
xoutRgb = pipeline.create(dai.node.XLinkOut)

# STREAMS
xinFrame.setStreamName("inFrame")
nnOut.setStreamName("nn")
xoutRgb.setStreamName("rgb")

# Properties
nn.setBlobPath(nnPath)
nn.setConfidenceThreshold(confidenceThreshold + 12)
nn.setNumClasses(classes)
nn.setCoordinateSize(coordinates)
nn.setAnchors(anchors)
nn.setAnchorMasks(anchorMasks)
nn.setIouThreshold(iouThreshold)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(True)

# Linking
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)
nn.passthrough.link(xoutRgb.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    def displayFrame(name, frame, detections):
        color = (255, 0, 255)

        if detections is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                #cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 2, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        while True:
            if cv2.waitKey(1) == ord('q'):
                break

    for file in [f for f in os.listdir(mediaPath) if os.path.isfile(os.path.join(mediaPath,f)) and f.endswith(".jpg") or f.endswith(".png")]:
        # Input queue will be used to send video frames to the device.
        qIn = device.getInputQueue(name="inFrame")
        # Output queue will be used to get nn data from the video frames.
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


        print(f"Using image...{os.path.join(mediaPath,file)}")
        frame = cv2.imread(os.path.join(mediaPath,file), cv2.IMREAD_COLOR)
                    
        # Get the dimensions of the image
        height, width, _ = frame.shape

        # Check if the image is a square
        if height == width:
            # The image is already a square, so we don't need to do anything
            pass
        else:
            # The image is not a square, so we need to crop it

            # Determine which dimension is larger (height or width)
            larger_dim = max(height, width)

            # Crop the image to a square with the larger dimension
            if height > width:
                # Crop the image along the height
                frame = frame[:width, :]
            else:
                # Crop the image along the width
                frame = frame[:, :height]

    
        cv2.namedWindow("rgb", cv2.WINDOW_NORMAL) 

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (nn_shape, nn_shape)))
        img.setTimestamp(monotonic())
        img.setWidth(nn_shape)
        img.setHeight(nn_shape)
        qIn.send(img)

        inRgb = qRgb.get()
        inDet = qDet.get()


        if inDet is not None:
            detections = inDet.detections
            print("DETECTIONS:")
            print(detections)

        if frame is not None:
            displayFrame("rgb", frame, detections)
