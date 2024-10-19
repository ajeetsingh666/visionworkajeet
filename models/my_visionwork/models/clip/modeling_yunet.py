import cv2

class FaceDetectorYuNet:
    def __init__(self, input_size=(320, 180), score_threshold=0.6):
        self.input_size = input_size
        model_path = "/home/ajeet/codework/ujjawal_github/visionwork/onnx/face_detection_yunet_2023mar.onnx"
        self.face_detector = cv2.FaceDetectorYN_create(model_path, "", input_size, score_threshold=0.6)

    def detect(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image from {image_path}")

        # Resize image to model input size
        resized_image = cv2.resize(image, self.input_size)

        # Perform face detection
        _, faces = self.face_detector.detect(resized_image)
        print(faces)

        # Return detected faces in a dictionary format
        return {
            "detected_faces": faces if faces is not None else [],
            "num_faces": len(faces) if faces is not None else 0
        }

    def classify(self, image_path):
        result = self.detect(image_path)
        num_faces = result["num_faces"]
        prob = 1  # Confidence for classification is assumed to be 1 for now

        # Classify based on the number of detected faces
        if num_faces == 0:
            classification = "no_person"
        elif num_faces == 1:
            classification = "single_person"
        elif num_faces >= 2:
            classification = "multiple_persons"
        else:
            classification = "Uncertain"

        return classification, prob

if __name__ == "__main__":
    # model_path = "/home/ajeet/codework/ujjawal_github/visionwork/onnx/face_detection_yunet_2023mar.onnx"
    face_detector = FaceDetectorYuNet(input_size=(320, 180))

    # Classify an image based on the number of detected faces
    image_path = "/tmp/video_incidents_ajeet/2665003/3_660.jpg"
    classification, probability = face_detector.classify(image_path)
    print(f"Classification: {classification}, Probability: {probability}")
