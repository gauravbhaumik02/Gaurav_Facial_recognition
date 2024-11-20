from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model

######pathsandvairables#########


# These lines define variables and initialize objects:
# face_data: Path to the directory containing face images.
# required_shape: Desired shape for the input face images (160x160 pixels).
# face_encoder: Instance of the InceptionResNetV2 model for face encoding.
# path: Path to the pre-trained weights file for the FaceNet model.
# face_detector: Instance of the MTCNN face detector.
# encodes: List to store encoded feature vectors of faces.
# encoding_dict: Dictionary to store encoding vectors mapped to face names.
# l2_normalizer: Instance of the L2 normalizer for feature vector normalization.
# python


face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################

# This function normalize takes an image as input, calculates its mean and standard deviation, and then normalizes the image using these statistics.
                    

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

# This block of code iterates over each directory containing face images (face_data) and processes each image:
# It reads the image using OpenCV, converts it to RGB format, and detects faces using MTCNN.
# For each detected face, it extracts the bounding box coordinates, crops the face region, and normalizes and resizes the face image to the required shape.
# It then encodes the face using the FaceNet model and stores the encoding vector in the encodes list.
# After processing all images for a particular person, it calculates the mean encoding vector, normalizes it using L2 normalization, and stores it in the encoding_dict dictionary, mapped to the person's name.


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        x = face_detector.detect_faces(img_RGB)
        x1, y1, width, height = x[0]['box']
        x1, y1 = abs(x1) , abs(y1)
        x2, y2 = x1+width , y1+height
        face = img_RGB[y1:y2 , x1:x2]
        
        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode
        
        
        
# Finally, this code saves the encoding dictionary (encoding_dict) to a pickle file for future use.
# Overall, this script processes face images, encodes them using FaceNet, and saves the encoding vectors along with their respective labels for use in facial recognition tasks.

    
path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)






