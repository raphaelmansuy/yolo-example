from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

# Visualize the results on the image
annotated_image = results[0].plot()

# Get the detected object categories
categories = results[0].names

# Display the annotated image with the tracker box and object categories
cv2.imshow("Detected Objects", annotated_image)
print("Detected Categories:", categories)
cv2.waitKey(0)
cv2.destroyAllWindows()
