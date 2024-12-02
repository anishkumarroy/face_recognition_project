import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_images_from_directory(directory):
    faces = []
    labels = []
    name_dict = {}
    label = 0

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_detected:
                face = gray[y:y+h, x:x+w]
                faces.append(face)
                labels.append(label)
                name_dict[label] = filename.split('.')[0]  # Store name without extension
            label += 1

    return faces, labels, name_dict

# Function to train the LBP recognizer
def train_recognizer(faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer

def recognize_faces(input_image, recognizer, name_dict):
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)
        name = name_dict.get(label, "Unknown")
        cv2.rectangle(input_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(input_image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return input_image

directory = 'face_repository'  # Directory containing face images
faces, labels, name_dict = load_images_from_directory(directory)
recognizer = train_recognizer(faces, labels)

input_image = cv2.imread('group_photo.jpg')  # Replace with your input image path
output_image = recognize_faces(input_image, recognizer, name_dict)

cv2.imshow('Recognized Faces', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()