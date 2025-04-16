import cv2
import numpy as np
import os
from PIL import Image

data_path = 'faces/'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

for file in os.listdir(data_path):
    if file.endswith('.jpg'):
        img_path = os.path.join(data_path, file)
        gray_img = Image.open(img_path).convert('L')
        img_np = np.array(gray_img, 'uint8')
        user_id = int(file.split('_')[1])
        faces.append(img_np)
        labels.append(user_id)

face_recognizer.train(faces, np.array(labels))
face_recognizer.save('face_model.yml')
print("Modelo entrenado y guardado.")
