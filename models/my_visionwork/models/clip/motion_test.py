# import cv2
# import numpy as np
# from ultralytics import YOLO  # Import the YOLO class

# print("hi")
# prev_centers = []
# speeds = []
# directions = []
# accelerations = []
# count = 0

# class PersonDetectorYOLO:
#     def __init__(self, model_path="yolov8n.pt", device="cpu"):
#         self.device = "cpu"
#         self.model = YOLO(model_path).to(self.device)
#         self.classes_to_detect = [0]  # Class 0 for "person" in COCO dataset

#     def predict(self, frame, conf_threshold=0.5):
#         # Perform inference
#         results = self.model.predict(source=frame, conf=conf_threshold, device=self.device, classes=self.classes_to_detect)
#         return results[0]

# # Instantiate the person detector
# detector = PersonDetectorYOLO(model_path="yolov8n.pt")

# # cap = cv2.VideoCapture('/home/ajeet/codework/yolo_testing/test3/2538482/2538482')
# # cap = cv2.VideoCapture('/home/ajeet/codework/yolo_testing/test5/2658010/2658010')
# cap = cv2.VideoCapture('/home/ajeet/codework/yolo_testing/test5/2595728/2595728')

# tracker = None
# tracked_person_bbox = None

# backSub = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if tracker is None:
#         # Use the YOLO model to detect people
#         detections = detector.predict(frame)
#         print("detection",detections)
#         print("inside_yolo")

#         if detections.boxes:
#             # Extract the first person detected
#             detection = detections.boxes[0]
#             x, y, w, h = detection.xywh[0].int().tolist()
            
#             w = int(w * 1.1)
#             h = int(h * 1.1)
#             tracked_person_bbox = (x - w // 2, y - h // 2, w, h)

#             # Initialize the tracker
#             tracker = cv2.TrackerKCF_create()
#             # tracker.create()
#             print("tracker", tracker)
#             tracker.init(frame, tracked_person_bbox)

#     else:
#         success, tracked_person_bbox = tracker.update(frame)

#         if success:
#             (x, y, w, h) = [int(v) for v in tracked_person_bbox]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             # Create a mask for motion detection outside the tracked bounding box
#             mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#             mask[y:y + h, x:x + w] = 255
#             inverse_mask = cv2.bitwise_not(mask)

#             fg_mask = backSub.apply(frame)
#             monitored_area = cv2.bitwise_and(fg_mask, fg_mask, mask=inverse_mask)

#             contours, _ = cv2.findContours(monitored_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             # for contour in contours:
#             #     if cv2.contourArea(contour) > 2000:
#             #         print("Motion detected outside tracked area!")
#             #         cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

#             largest_contour = None
#             largest_area = 0

#             for contour in contours:
#                 area = cv2.contourArea(contour)
#                 # print(area)

#                 if area < 50: 
#                     continue

#                 buffer = 5
#                 cx, cy, cw, ch = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)  # Draw rectangle in blue
#                 if (cx + cw + buffer >= x) and (cx <= x + w + buffer) and (cy + ch + buffer >= y) and (cy <= y + h + buffer):
#                     # print("Motion detected inside the tracked area! Ignored.")
#                     continue  # Skip this contour if it's inside the tracked area
                
#                 if area > largest_area:
#                     print()
#                     largest_area = area
#                     largest_contour = contour

#             # print(largest_area)
#             if largest_contour is not None:
#                 # (x, y, w, h) = cv2.boundingRect(largest_contour)
#                 # center = (x + w // 2, y + h // 2)

#                 # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                 rect = cv2.minAreaRect(largest_contour)
#                 box = cv2.boxPoints(rect)
#                 box = np.int0(box)

#                 cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

#                 center = (int(rect[0][0]), int(rect[0][1]))

#                 # cx, cy, cw, ch = cv2.boundingRect(largest_contour)
#                 # if (cx + cw >= x) and (cx <= x + w) and (cy + ch >= y) and (cy <= y + h):
#                 #     print("Motion detected inside the tracked area! Ignored.")
#                 #     continue  # Skip this contour if it's inside the tracked area

#                 if len(prev_centers) > 0:
#                     dx = center[0] - prev_centers[-1][0]
#                     dy = center[1] - prev_centers[-1][1]
#                     distance = np.sqrt(dx ** 2 + dy ** 2)

#                     speed = distance 
#                     speeds.append(speed)
#                     # print(f"Speed: {speed:.2f} pixels/frame", end='')

#                     # Calculate direction (in radians)
#                     # direction = np.arctan2(dy, dx)
#                     # directions.append(direction)

#                     # Calculate acceleration
#                     # if len(speeds) > 1:
#                     #     acceleration = speeds[-1] - speeds[-2]  # Change in speed
#                     #     accelerations.append(acceleration)

#                     # print(f"Speed: {speed:.2f} pixels/frame, Direction: {direction:.2f} radians", end='')
#                     # if len(accelerations) > 0:
#                     #     print(f", Acceleration: {accelerations[-1]:.2f} pixels/frame²")
#                     # else:
#                     #     print(", Acceleration: N/A")

#                 prev_centers.append(center)

#                 if speeds and speeds[-1] > 10:
#                     print(speeds[-1])
#                     count = count + 1
#                     print(f"{count}: Violation: Second person detected in the background.")

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


###################################


import cv2
import numpy as np
from ultralytics import YOLO  # Import the YOLO class

print("hi")
prev_centers = []
speeds = []
count = 0

class PersonDetectorYOLO:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.classes_to_detect = [0]  # Class 0 for "person" in COCO dataset

    def predict(self, frame, conf_threshold=0.5):
        # Perform inference
        results = self.model.predict(source=frame, conf=0.5, device=self.device, classes=self.classes_to_detect)
        return results[0]

# Instantiate the person detector
detector = PersonDetectorYOLO(model_path="yolov8n.pt")

# cap = cv2.VideoCapture('/home/ajeet/codework/yolo_testing/test5/2565397/2565397')
# cap = cv2.VideoCapture('/home/ajeet/codework/yolo_testing/test6/2628150/2628150')
cap = cv2.VideoCapture('/home/ajeet/codework/ujjawal_github/Dataset/benchmark_dataset_13_1_20/2570949/2570949')
backSub = cv2.createBackgroundSubtractorMOG2()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Calculate the interval to process frames at 1 FPS

frame_count = 0

frame_no = 0
while True:
    frame_no = frame_no + 1
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    print(f"frame_count: {frame_count}")

    # Use the YOLO model to detect people
    detections = detector.predict(frame)
    # print("detection", detections)

    if detections.boxes:
        # Extract the first person detected
        detection = detections.boxes[0]
        x, y, w, h = detection.xywh[0].int().tolist()

        
        # Increase the size of the bounding box
        w = int(w * 1.2)
        h = int(h * 1.2)
        tracked_person_bbox = (x - w // 2, y - h // 2, w, h)

        # Draw the bounding box for the detected person
        cv2.rectangle(frame, (tracked_person_bbox[0], tracked_person_bbox[1]), 
                      (tracked_person_bbox[0] + tracked_person_bbox[2], tracked_person_bbox[1] + tracked_person_bbox[3]), 
                      (255, 255, 255), thickness=cv2.FILLED)

        # Create a mask for motion detection outside the detected bounding box
        # mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        # mask[tracked_person_bbox[1]:tracked_person_bbox[1] + tracked_person_bbox[3], 
        #      tracked_person_bbox[0]:tracked_person_bbox[0] + tracked_person_bbox[2]] = 255
        # inverse_mask = cv2.bitwise_not(mask)

        # fg_mask = backSub.apply(frame)


        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[tracked_person_bbox[1]:tracked_person_bbox[1] + tracked_person_bbox[3], 
             tracked_person_bbox[0]:tracked_person_bbox[0] + tracked_person_bbox[2]] = 255
        inverse_mask = cv2.bitwise_not(mask)

        fg_mask = backSub.apply(frame)
        monitored_area = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)


        cv2.imshow('fg_mask', fg_mask)
        cv2.imshow('Actual Frame', frame)

        contours, _ = cv2.findContours(monitored_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        largest_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            print("Area", area)

            if area < 1: 
                continue

            buffer = 1
            cx, cy, cw, ch = cv2.boundingRect(contour)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)  # Draw rectangle in blue
            if (cx + cw + buffer >= x) and (cx <= x + w + buffer) and (cy + ch + buffer >= y) and (cy <= y + h + buffer):
                print("Motion detected inside the tracked area! Ignored.")
                continue  # Skip this contour if it's inside the tracked area

            if area > largest_area:
                largest_area = area
                largest_contour = contour

        if largest_contour is not None:
            count += 1
            print(f"{count}: Violation: Second person detected in the background.")
            # rect = cv2.minAreaRect(largest_contour)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)

            # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # center = (int(rect[0][0]), int(rect[0][1]))

            # if len(prev_centers) > 0:
            #     dx = center[0] - prev_centers[-1][0]
            #     dy = center[1] - prev_centers[-1][1]
            #     distance = np.sqrt(dx ** 2 + dy ** 2)

            #     speed = distance 
            #     speeds.append(speed)

            # prev_centers.append(center)

            # # Check for speed violations
            # if speeds and speeds[-1] > 1:
            #     count += 1
            #     print(f"{count}: Violation: Second person detected in the background.")

    # cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#########################

# import cv2
# import numpy as np
# from ultralytics import YOLO  # Import the YOLO class

# class PersonDetectorYOLO:
#     def __init__(self, model_path="yolov8n.pt", device="cpu"):
#         self.device = "cpu"
#         self.model = YOLO(model_path).to(self.device)
#         self.classes_to_detect = [0]  # Class 0 for "person" in COCO dataset

#     def predict(self, frame, conf_threshold=0.5):
#         # Perform inference
#         results = self.model.predict(source=frame, conf=conf_threshold, device=self.device, classes=self.classes_to_detect)
#         return results[0]

# # Instantiate the person detector
# detector = PersonDetectorYOLO(model_path="yolov8n.pt")

# cap = cv2.VideoCapture('/tmp/video_incidents_ajeet/output.mp4')
# backSub = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Use the YOLO model to detect people
#     detections = detector.predict(frame)

#     if detections.boxes:
#         # Extract the first person detected
#         detection = detections.boxes[0]
#         x, y, w, h = detection.xywh[0].int().tolist()
        
#         # Create a mask for the tracked person's bounding box
#         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#         mask[y:y + h, x:x + w] = 255  # Fill the area of the bounding box
#         inverse_mask = cv2.bitwise_not(mask)

#         # Apply background subtraction
#         fg_mask = backSub.apply(frame)

#         # Only keep the motion outside the tracked bounding box
#         monitored_area = cv2.bitwise_and(fg_mask, fg_mask, mask=inverse_mask)

#         # Find contours
#         contours, _ = cv2.findContours(monitored_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Draw the bounding box of the tracked person
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Check for contours and handle them
#         for contour in contours:
#             area = cv2.contourArea(contour)

#             if area < 200:  # Adjust this threshold as needed
#                 continue
            
#             # Get bounding box for the contour
#             cx, cy, cw, ch = cv2.boundingRect(contour)

#             # Check if the contour intersects with the tracked bounding box
#             if (cx + cw >= x) and (cx <= x + w) and (cy + ch >= y) and (cy <= y + h):
#                 print("Motion detected inside the tracked area! Ignored.")
#                 continue

#             # Draw the contour if it’s outside the bounding box
#             cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
