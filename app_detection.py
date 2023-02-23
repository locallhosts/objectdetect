import cv2
import os
import numpy as np

object_dir = os.path.abspath('objectdetect/')

# Load YOLOv3 configuration and weights
config_path = os.path.join(object_dir, 'tensor', 'yolov3.cfg')
config_paths = os.path.join(object_dir, 'tensor', 'yolov3.weights')

net = cv2.dnn.readNetFromDarknet(config_path, config_paths)

# Load image and prepare it for object detection
cap = cv2.VideoCapture(0)

# Define labels and colors
LABELS = open(os.path.join(object_dir, 'data', 'coco.names')).read().strip().split('\n')
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
CONFIDENCE_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: failed to capture frame")
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set input to the loaded network and perform a forward pass to detect objects
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # Process the layer outputs to extract detected objects and their properties
    confidence_threshold = 0.5
    nms_threshold = 0.4
    class_ids = []
    confidences = []
    boxes = []
    (H, W) = frame.shape[:2]

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maxima suppression to eliminate redundant overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Draw the boxes and class labels on the image
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 8)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the results, or  Show the image with object labels
    cv2.imshow('Image', frame)
  
 # Wait for a key press, and if the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
