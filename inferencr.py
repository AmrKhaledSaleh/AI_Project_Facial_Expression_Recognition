import numpy as np
# from keras.preprocessing.image import load_img, img_to_array
import cv2
from keras.models import model_from_json


# Loading my Trained Model
with open('models/FER/cnn_1/model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

expression_detector = model_from_json(loaded_model_json)
expression_detector.load_weights('models/FER/cnn_1/weights.h5')

# using ready trained haarcascades model from openVC to detect face
face_detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


label_dict = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a VideoCapture object, where 0 corresponds to the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the window name
window_name = 'VideoCapture'

# Define the desired width and height for the display window


# Create a named window
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(window_name, 800, 750)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    

    for (x, y, w, h) in faces:
        #Extract the region of interest (ROI)
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Expand dimensions to make it compatible with model input shape (add batch dimension)
        img = np.expand_dims(np.expand_dims(cv2.resize(face_roi, (48, 48)), -1), 0)

        # Make a prediction using my model
        expression_prediction = expression_detector.predict(img)

        # Get the corresponding emotion label from the dictionary
        emotion_label = label_dict[np.argmax(expression_prediction)]

        #Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (250, 13, 12), thickness=3)

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
    cv2.imshow(window_name, frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()