import cv2
import pandas as pd
import os
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone
import numpy as np
# Initialize YOLO model and PaddleOCR
model = YOLO('best_float32.tflite')
ocr = PaddleOCR()


# Load class list from file
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Open video file
cap = cv2.VideoCapture('tc.mp4')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
#        print(results)
        for result in results[0]:
            print(result)
            text = result[1][0]
            detected_text.append(text)
      
    # Join all detected texts into a single string
    return ''.join(detected_text)

count = 0
#area=[(2,95),(1,135),(1018,146),(1016,125)]
cy1=102
offset=20
while True:
    ret, frame = cap.read()
#    count += 1
#    if count % 2 != 0:
#        continue

    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection
    results = model.predict(frame, imgsz=240)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
#        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
#        if result>=0:
        if cy1<(cy+offset) and cy1>(cy-offset):
#           cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0, 0), 2)
           crop = frame[y1:y2, x1:x2]
           crop=cv2.resize(crop,(210,50))
           text = perform_ocr(crop)
           text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','').replace('-',' ')
#            print(text)
           cvzone.putTextRect(frame, f'{text}', (cx - 50, cy - 30), 1, 1)
         

           
#    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,255),2)
    cv2.line(frame,(3,102),(1018,102),(0,0,255),1)
    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
