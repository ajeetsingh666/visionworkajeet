import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
import os

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8:

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        x1, y1, w, h = box

        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, image):
        # self.img = cv2.imread(self.input_image)
        self.img = cv2.imread(image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        phone_detected = False
        phone_score = [-1]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            # print(max_score)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                # print(f"class_id: {class_id}")

                if class_id == 2:
                    phone_detected = True
                    phone_score.append(max_score)
                    # Extract the bounding box coordinates from the current row
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # Calculate the scaled coordinates of the bounding box
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # Add the class ID, score, and box coordinates to the respective lists
                    class_ids.append(class_id)
                    scores.append(max_score)
                    boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image , phone_detected, max(phone_score)

    def main(self, image):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess(image)

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs) # output image

    # def classify(self, image):
    #     # detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx", image, 0.50, 0.40)

    #     # Perform object detection and obtain the output image
    #     output_image, phone_detected, phone_score = detection.main(image)
        
    #     # if phone_detected:
    #     #     print(f"{image}: phone_detected ")
    #     # else:
    #     #     print(f"{image}: phone_not_detected ")

    #     return phone_detected, phone_score



def save_image(image, path, image_name):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, image_name), image)





if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    # parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    # parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    # parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    # args = parser.parse_args()

    # Check the requirements and select the appropriate backend (CPU or GPU)
    # check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    # Create an instance of the YOLOv8 class with the specified arguments
    # detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)
    # images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]


    # images = [os.path.join("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged") if filename.startswith("0_")]


    # images = [os.path.join("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332") if filename.startswith("0_")]


    images = [os.path.join("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged" , filename) 
    for filename in os.listdir("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged") if filename.startswith("0_")]


    # images = images[:1]

    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    for image in images:
        detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx",image, 0.50, 0.40)

        # Perform object detection and obtain the output image
        output_image, phone_detected = detection.main()
        
        if phone_detected:
            print(f"{image}: phone_detected ")
        else:
            print(f"{image}: phone_not_detected ")



        save_path_with_phone = "/home/ajeet/codework/finetuned_yolo"
        image_name = os.path.basename(detection.input_image)
        save_image(output_image, save_path_with_phone, image_name)

        # Display the output image in a window
        # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        # cv2.imshow("Output", output_image)

        # # Wait for a key press to exit
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()