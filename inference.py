import cv2
import numpy as np
from keras.models import model_from_json


class FER:
    def __init__(self, video_source=None, model_type='cnn'):
        self.face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        # Create a VideoCapture object
        self.cap = cv2.VideoCapture(video_source) if video_source else cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            exit()

        # Define the window name
        self.window_name = 'VideoCapture'

        # Load the specified model
        if model_type == 'cnn':
            self.expression_detector = self.load_cnn_model()
        elif model_type == 'ann':
            self.expression_detector = self.load_ann_model()
        else:
            raise ValueError("Invalid model type. Use 'cnn' or 'ann'.")

        # Create a named window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 750)

    @staticmethod
    def load_cnn_model():
        # Loading CNN Trained Model
        cnn_model_path = 'models/FER/cnn/model.json'
        cnn_weights_path = 'models/FER/cnn/weights.h5'

        with open(cnn_model_path, 'r') as json_file:
            cnn_model_json = json_file.read()

        cnn_model = model_from_json(cnn_model_json)
        cnn_model.load_weights(cnn_weights_path)

        return cnn_model

    @staticmethod
    def load_ann_model():
        # Loading ANN Trained Model
        ann_model_path = 'models/FER/ann/model.json'
        ann_weights_path = 'models/FER/ann/weights.h5'

        with open(ann_model_path, 'r') as json_file:
            ann_model_json = json_file.read()

        ann_model = model_from_json(ann_model_json)
        ann_model.load_weights(ann_weights_path)

        return ann_model

    def process_frame(self, frame):
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI)
            face_roi = gray_frame[y:y + h, x:x + w]

            # Expand dimensions to make it compatible with model input shape (add batch dimension)
            img = np.expand_dims(np.expand_dims(cv2.resize(face_roi, (48, 48)), -1), 0)

            # Get the corresponding emotion label from the dictionary
            emotion_label = self.image_predict(img=img)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 13, 12), thickness=3)

            # Display the predicted expression
            img = frame
            txt = f"Expression: {emotion_label}"
            position = (x, y - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            txt_color = (0, 0, 0)
            outline_color = (255, 255, 255)
            cv2.putText(img, txt, position, font, font_scale, outline_color, 5)
            cv2.putText(img, txt, position, font, font_scale, txt_color, 2)

        # Display the frame in a window
        cv2.imshow(self.window_name, frame)

    def image_predict(self, img, img_path='0'):
        if img_path != '0':
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(np.expand_dims(cv2.resize(img, (48, 48)), -1), 0)

        # Make a prediction using the model
        expression_prediction = self.expression_detector.predict(img)

        # Get the corresponding emotion label from the dictionary
        label = self.label_dict[np.argmax(expression_prediction)]

        return label

    def run(self):
        while True:
            # Read a frame from the camera
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            self.process_frame(frame)

            # Break the loop if the 'q' key is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 'q' key or 'Esc' key
                break

        # Release the VideoCapture object and close the window
        self.cap.release()
        cv2.destroyAllWindows()

# Example usage:
# detector_cnn = FER(model_type='cnn')  # For CNN model
# detector_ann = FaceExpressionDetector(model_type='ann')  # For ANN model
# detector_cnn.run()
# detector_ann.run()
