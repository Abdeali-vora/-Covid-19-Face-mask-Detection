import numpy
import os
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

model =  load_model('model(2).h5')

webcam = cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face = trained_face_data.detectMultiScale(gray)
    faces=[]
    preds=[]
    for (x,y,w,h) in face:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame,(224,224))
        face_frame = img_to_array(face_frame)
        face_frame = numpy.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces.append(face_frame)
        if len(faces)>0:
            preds = model.predict(faces)
            for pred in preds:
                (With_Mask,Without_mask)=pred
                if With_Mask > Without_mask:
                    label = 'With_Mask'
                    color=(0,255,0)
                else:
                    label='Without_Mask'
                    color=(0,0,255)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
                cv2.putText(frame, label , (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.imshow("hey",frame)
    key = cv2.waitKey(1)
    if key==65 or key==97:
       break      
webcam.release()
cv2.destroyAllWindows()


