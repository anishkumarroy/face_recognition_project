

import streamlit as st
import face_recognition as fr
import os
import cv2
import numpy as np
import face_recognition
from PIL import Image

img_dir = "temp"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./face_repository"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file("face_repository/" + f)
                encodings = fr.face_encodings(face)
                if encodings:
                    encoding = encodings[0]
                    encoded[f.split(".")[0]] = encoding
    return encoded

def classify_face(im, tolerance=0.5):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = fr.compare_faces(faces_encoded, face_encoding, tolerance=tolerance)
        name = "Unknown"

        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(img, (left-20, top-10), (right+20, bottom+15), (300, 0, 0), 2)
            cv2.rectangle(img, (left-20, bottom -10), (right+20, bottom+30), (500, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -10, bottom + 20), font, 0.75, (255, 255, 255), 2)

    max_width = 1280
    max_height = 960
    height, width = img.shape[:2]
    scale_ratio = min(max_width / width, max_height / height)
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)
    img_resized = cv2.resize(img, (new_width, new_height))

    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

    return img_pil, face_names

st.title("Face Recognition App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img_path = os.path.join("temp", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img_resized, face_names = classify_face(img_path)

    st.image(img_resized, caption="Processed Image", use_container_width=True)
    st.write("Recognized Faces:", ", ".join(face_names))