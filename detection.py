import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:/Users/datab/Desktop/detection/yolov3.weights", "C:/Users/datab/Desktop/detection/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("C:/Users/datab/Desktop/detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    # Capture each frame from the camera
    ret, frame = cap.read()

    # Get the height, width, and channels of the frame
    height, width, channels = frame.shape

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store detection results
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection in the outputs
    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    # Get the coordinates of the box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Max Suppression to avoid multiple boxes on the same object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # Display the frame with detection
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
# import cv2Q
# import numpy as np

# # Load YOLO
# net = cv2.dnn.readNet(r"C:\Users\datab\Desktop\detection\yolov3.weights", r"C:\Users\datab\Desktop\detection\yolov3.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# # Load COCO class labels
# with open(r"C:\Users\datab\Desktop\detection\coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Initialize the webcam
# cap = cv2.VideoCapture(0)  # 0 is the default camera

# while True:
#     # Capture each frame from the camera
#     ret, frame = cap.read()

#     # Get the height, width, and channels of the frame
#     height, width, channels = frame.shape

#     # Prepare the frame for YOLO input
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Initialize lists to store detection results
#     class_ids = []
#     confidences = []
#     boxes = []

#     # Loop over each detection in the outputs
#     for out in outs:
#         for detection in out:
#             for obj in detection:
#                 scores = obj[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.5:
#                     # Object detected
#                     center_x = int(obj[0] * width)
#                     center_y = int(obj[1] * height)
#                     w = int(obj[2] * width)
#                     h = int(obj[3] * height)

#                     # Get the coordinates of the box
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#     # Apply Non-Max Suppression to avoid multiple boxes on the same object
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels
#     font = cv2.FONT_HERSHEY_PLAIN
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             color = (0, 255, 0)  # Green color for bounding boxes
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

#     # Display the frame with detection
#     cv2.imshow("YOLO Object Detection", frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()
