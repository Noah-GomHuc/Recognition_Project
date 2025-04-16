import os, cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vidCapture = cv2.VideoCapture(0)

def extract_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is None:
        return None
    for (x, y, w, h) in faces:
        return gray[y:y+h, x:x+w]

user_id = input("Introducir un ID numérico para el usuario: ")

count = 0
os.makedirs("faces", exist_ok=True)

while True:
    ret, frame = vidCapture.read()
    if extract_face(frame) is not None:
        count += 1
        face = cv2.resize(extract_face(frame), (200, 200))
        file_name_path = f'faces/user_{user_id}_{count}.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.imshow('capturando rostros...', frame)
    else:
        print("no se te encontro fantasma...")
    
    if cv2.waitKey(10) & 0xFF == ord("q"):  
        break

vidCapture.release()
cv2.destroyAllWindows()
print("Finish")
