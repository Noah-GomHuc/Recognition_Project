#Libs
import cv2, numpy, os, argparse

#built in haarscascade_frontalface...
'''cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')'''

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

vidcapture = cv2.VideoCapture(0)

def detect_box(vidcapture):
    gray_image = cv2.cvtColor(vidcapture, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40,40))
    for (i, j, k, l) in faces:
        cv2.rectangle(vidcapture, (i, j), (i + k, j + l), (0, 255, 0), 4)
        roi_gray = gray_image[j:j + l, i:i + k]
        roi_color = vidcapture[j:j + l, i:i + k]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        eyes = eye_classifier.detectMultiScale(gray_image)
        for (a, b, c, d) in eyes:
            cv2.rectangle(vidcapture, (a, b), (a + c, b + d), (255, 0, 0), 2)
    return faces, eyes
    
flag = True
while flag == True:
    result, vid_frame = vidcapture.read()
    if result == False:
        flag = False
    faces, eyes = detect_box(vid_frame)
    cv2.imshow(
        "Detection...", vid_frame
    )
    print(f"Caras detectadas: {len(faces)} | Ojos detectados: {len(eyes)}")
    if cv2.waitKey(10) & 0xFF == ord("q"):
        flag = False
vidcapture.release()
cv2.destroyAllWindows()