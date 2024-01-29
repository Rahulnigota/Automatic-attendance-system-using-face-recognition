import face_recognition
import cv2
import numpy  as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)




ankit_image = face_recognition.load_image_file("E:\Coding(VS)\python projects\Face_recogination\image/ankit.png")
ankit_encoding = face_recognition.face_encodings(ankit_image)[0]

rahuls_image = face_recognition.load_image_file("E:\Coding(VS)\python projects\Face_recogination\image/rahul.jpg")
rahuls_encoding = face_recognition.face_encodings(rahuls_image)[0]


rohit_image = face_recognition.load_image_file("E:\Coding(VS)\python projects\Face_recogination\image/rohit.png")
rohit_encoding = face_recognition.face_encodings(rohit_image)[0]





known_face_encoding = [
    
   
    ankit_encoding,
    rahuls_encoding,
    rohit_encoding
]

known_faces_names = [ 
    "rohit",
    "ankit",
    "Rahul Singh Nigota",
    "Rohit Kumar Kain",
    "Shivaka Baghel"
]


student = known_faces_names.copy()

face_location = []
face_encoding = []
faces_names = []
s=True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")



f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_location = face_recognition.face_locations(rgb_small_frame)
        face_encoding = face_recognition.face_encodings(rgb_small_frame,face_location)
        faces_names = []
        for face_encoding in face_encoding:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name =""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_matche_index = np.argmin(face_distance)
            if matches[best_matche_index]:
                name = known_faces_names[best_matche_index]

            faces_names.append(name)
            if name in known_faces_names:
                if name in student:
                    student.remove(name)
                    print(student)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([current_time,name])
    cv2.imshow("Attendance system",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()