import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import os
import platform

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Load input video
input_path = "video_keys_new.mp4"
cap = cv2.VideoCapture(input_path)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_with_boxes.mp4", fourcc, fps, (width, height))

# Movement tracking
last_positions = {}
movement_status = defaultdict(lambda: {"moving": False, "last_moved_time": 0})

movement_threshold = 15  # pixels
stable_duration_required = 1.0  # seconds
font = cv2.FONT_HERSHEY_SIMPLEX

# Class color mapping
class_colors = {}

# Center of a box
def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)[0]
    current_positions = {}

    for box in results.boxes.data:
        x1, y1, x2, y2, score, class_id = box.tolist()
        class_id = int(class_id)
        name = model.names[class_id]
        center = get_center((x1, y1, x2, y2))
        current_positions[name] = center

        # Assign a consistent color for this class
        if name not in class_colors:
            np.random.seed(hash(name) % 2**32)
            class_colors[name] = tuple(np.random.randint(0, 255, size=3).tolist())

        color = class_colors[name]

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label with confidence score
        label = f"{name} {score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), font, 0.9, color, 2)

    # Get current time for stabilization logic
    current_time = time.time()

    for name, current_center in current_positions.items():
        if name in last_positions:
            prev_center = last_positions[name]
            distance = np.linalg.norm(np.array(current_center) - np.array(prev_center))

            if distance > movement_threshold:
                # Mark as moving and record movement time
                movement_status[name]["moving"] = True
                movement_status[name]["last_moved_time"] = current_time
            else:
                if movement_status[name]["moving"]:
                    time_since_move = current_time - movement_status[name]["last_moved_time"]
                    if time_since_move > stable_duration_required:
                        video_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        print(f'✅ Movement Detected: "{name}" stabilized at {video_seconds:.2f}s (after movement)')
                        movement_status[name]["moving"] = False

        # Update last position
        last_positions[name] = current_center

    # Write and show the frame
    out.write(frame)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
