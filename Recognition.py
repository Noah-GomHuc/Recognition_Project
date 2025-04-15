#Libs
import cv2, numpy, os, argparse

#built in haarscascade_frontalface...
'''cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')'''

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

vidcapture = cv2.VideoCapture(0)

def detect_box(vidcapture):
    gray_image = cv2.cvtColor(vidcapture, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40,40))
    for (i, j, k, l) in faces:
        cv2.rectangle(vidcapture, (i, j), (i + k, j + l), (0, 255, 0), 4)
        return faces
flag = True
while flag == True:
    result, vid_frame = vidcapture.read()
    if result == False:
        flag = False
    
    faces = detect_box(vid_frame)
    cv2.imshow(
        "Detection...", vid_frame
    )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        flag = False
vidcapture.release()
cv2.destroyAllWindows()