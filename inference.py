import cv2
import numpy as np
from keras.models import model_from_json


class FER:
    def __init__(self, model_type='cnn'):
        self.face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.label_dict = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'neutral', 5: 'sad', 6: 'surprise'}

        # Load the specified model
        if model_type == 'cnn':
            self.expression_detector = self.load_cnn_model()
        elif model_type == 'ann':
            self.expression_detector = self.load_ann_model()
        else:
            raise ValueError("Invalid model type. Use 'cnn' or 'ann'.")

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
            expression_prediction = self.expression_detector.predict(img)

            # Get the corresponding emotion label from the dictionary
            emotion_label = self.label_dict[np.argmax(expression_prediction)]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 13, 12), thickness=3)

            # Display the predicted expression
            img = frame
            txt = f"Expression: {emotion_label}"
            position = (x, y + 20)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            txt_color = (0, 0, 0)
            outline_color = (255, 255, 255)
            cv2.putText(img, txt, position, font, font_scale, outline_color, 5)
            cv2.putText(img, txt, position, font, font_scale, txt_color, 2)

        return frame

    def image_use(self, img_path):
        img = cv2.imread(img_path)
        edited_img = self.process_frame(img)

        window_name = 'Labeled Image'

        # Create a named window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 850, 600)

        # Display the image in a window
        cv2.imshow(window_name, edited_img)

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ved(self, t, video_source=None):
        # Define the window name
        window_name = 'Test throw Camera'

        # Create a VideoCapture object
        if t:
            cap = cv2.VideoCapture(video_source)
            window_name = 'Test throw video'
            if not cap.isOpened():
                print("Error: Could not open video source.")
                exit()
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open the camera.")
                exit()

        # Create a named window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 750)

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            frame = self.process_frame(frame)
            cv2.imshow(window_name, frame)

            # Break the loop if the 'q' key is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 'q' key or 'Esc' key
                break

        # Release the VideoCapture object and close the window
        cap.release()
        cv2.destroyAllWindows()
