import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Start video capture from webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam or change to video file path

# Function to draw bounding boxes and count people
def draw_bounding_boxes_and_count(frame, boxes, confidences, class_ids, indexes):
    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = f"{int(confidences[i] * 100)}%"
            color = (0, 255, 0)  # Green for person detection

            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence_label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "person":
                people_count += 1

    # Display people count on the frame
    cv2.putText(frame, f"People Count: {people_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame, people_count

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Get the height, width, and channels of the frame
    height, width, channels = frame.shape

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider person detection with confidence > 50%
            if confidence > 0.6 and classes[class_id] == "person":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

    # Draw bounding boxes and display people count
    frame, people_count = draw_bounding_boxes_and_count(frame, boxes, confidences, class_ids, indexes)

    # Display the frame with detection
    cv2.imshow("Person Detection with Counter", frame)

    # Press 'q' to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
