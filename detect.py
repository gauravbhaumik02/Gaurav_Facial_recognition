import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize,l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle

# These lines define threshold values and required size for face recognition:
# confidence_t: Confidence threshold for face detection using MTCNN.
# recognition_t: Recognition threshold for identifying faces based on cosine distance.
# required_size: Required size for input face images to be fed into the face encoder.

confidence_t=0.99
recognition_t=0.5
required_size = (160,160)

# This function get_face takes an image and bounding box coordinates of a detected face and extracts the face region from the image

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

# This function get_encode takes a face encoder model, a face image, and a desired size, and returns the encoded feature vector for the face.

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

# This function load_pickle loads a pickle file containing encoding vectors mapped to face names and returns the dictionary.

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

# This function detect takes an image, a face detector model, a face encoder model, and an encoding dictionary as input, and performs face detection and recognition:
# It first converts the image to RGB format and uses the face detector to detect faces.
# For each detected face, it extracts the face region, encodes it using the face encoder, and normalizes the encoding vector.
# It compares the normalized encoding vector with the encoding vectors in the encoding dictionary using cosine distance.
# If a matching face is found within the recognition threshold, it annotates the image with the recognized name and distance.
# If no match is found, it annotates the image as 'unknown'.

def detect(img ,detector,encoder,encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img 



# This block of code initializes the face encoder model, loads pre-trained weights, loads the encoding dictionary, and starts the webcam for real-time face recognition:
# It creates instances of the face encoder, face detector, and loads the encoding dictionary from the pickle file.
# It initializes the webcam using OpenCV and continuously reads frames from the webcam.
# For each frame, it calls the detect function to perform face detection and recognition.
# It displays the annotated frame with recognized faces in real-time until the user presses 'q' to quit.

if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret,frame = cap.read()

        if not ret:
            print("CAM NOT OPEND") 
            break
        
        frame= detect(frame , face_detector , face_encoder , encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    


