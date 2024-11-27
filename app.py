import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

# YOLO files path
person_weights_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/yolov3-obj_final.weights'
person_cfg_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/yolov3_pb.cfg'
helmet_weights_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/yolov3-helmet.weights'
helmet_cfg_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/yolov3-helmet.cfg'
person_labels_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/coco.names'
helmet_labels_path = '/Users/deepakgrover/Documents/Computer_Vision/activity_2/streamlit/helmet.names'

# Load YOLO models
person_net = cv2.dnn.readNetFromDarknet(person_cfg_path, person_weights_path)
helmet_net = cv2.dnn.readNetFromDarknet(helmet_cfg_path, helmet_weights_path)

# Load labels
person_labels = open(person_labels_path).read().strip().split('\n')
helmet_labels = open(helmet_labels_path).read().strip().split('\n')

# Streamlit App
st.title("Helmet and Person Detection App")

# Image upload widget
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Set detection thresholds
probability_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
nms_threshold = st.slider("NMS Threshold", 0.1, 1.0, 0.3)

if uploaded_image:
    # Convert the uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_image).convert('RGB'))
    image_input = image.copy()
    height, width = image_input.shape[:2]

    # Preprocess the image for YOLO
    blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Forward pass for person detection
    person_net.setInput(blob)
    person_layer_names = [person_net.getLayerNames()[int(i) - 1] for i in person_net.getUnconnectedOutLayers()]
    person_detections = person_net.forward(person_layer_names)

    # Forward pass for helmet detection
    helmet_net.setInput(blob)
    helmet_layer_names = [helmet_net.getLayerNames()[int(i) - 1] for i in helmet_net.getUnconnectedOutLayers()]
    helmet_detections = helmet_net.forward(helmet_layer_names)

    # Initialize lists for storing bounding box data
    person_boxes, person_confidences, person_class_ids = [], [], []
    helmet_boxes, helmet_confidences, helmet_class_ids = [], [], []

    # Process person detections
    for detection in person_detections:
        for result in detection:
            scores = result[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > probability_threshold:
                box = result[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                person_boxes.append([x, y, int(box_width), int(box_height)])
                person_confidences.append(float(confidence))
                person_class_ids.append(class_id)

    # Process helmet detections
    for detection in helmet_detections:
        for result in detection:
            scores = result[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > probability_threshold:
                box = result[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype("int")
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                helmet_boxes.append([x, y, int(box_width), int(box_height)])
                helmet_confidences.append(float(confidence))
                helmet_class_ids.append(class_id)

    # Apply non-maxima suppression
    person_indices = cv2.dnn.NMSBoxes(person_boxes, person_confidences, probability_threshold, nms_threshold)
    helmet_indices = cv2.dnn.NMSBoxes(helmet_boxes, helmet_confidences, probability_threshold, nms_threshold)

    # Generate random colors for labels
    np.random.seed(42)
    person_colors = np.random.randint(0, 255, size=(len(person_labels), 3), dtype="uint8")
    helmet_colors = np.random.randint(0, 255, size=(len(helmet_labels), 3), dtype="uint8")

    # Draw bounding boxes for persons
    if len(person_indices) > 0:
        for i in person_indices.flatten():
            x, y, w, h = person_boxes[i]
            color = [int(c) for c in person_colors[person_class_ids[i]]]
            cv2.rectangle(image_input, (x, y), (x + w, y + h), color, 2)
            text = f"{person_labels[person_class_ids[i]]}: {person_confidences[i]:.2f}"
            cv2.putText(image_input, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw bounding boxes for helmets
    if len(helmet_indices) > 0:
        for i in helmet_indices.flatten():
            x, y, w, h = helmet_boxes[i]
            color = [int(c) for c in helmet_colors[helmet_class_ids[i]]]
            cv2.rectangle(image_input, (x, y), (x + w, y + h), color, 2)
            text = f"{helmet_labels[helmet_class_ids[i]]}: {helmet_confidences[i]:.2f}"
            cv2.putText(image_input, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the final image with bounding boxes
    st.image(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
