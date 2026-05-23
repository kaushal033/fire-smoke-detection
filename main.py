from ultralytics import YOLO 
import cv2 
import requests 
import numpy as np 
import time 
import winsound 

model = YOLO("weights/fire_smoke_yolov8s.pt")

URL = "http://10.106.172.91:8080/shot.jpg" # Enter URL displayed in your IP Webcam app (displayed in the bottom)

while True:
    try:
        img_resp = requests.get(URL, timeout=5)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)

        if frame is None:
            print("⚠️ Could not decode frame, skipping...")
            continue
        results = model(frame, conf=0.4)
        annotated = results[0].plot()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])  
            conf = float(box.conf[0]) 
          
            if conf > 0.6:
                if cls_id == 0:
                    print("🔥 Fire detected!")
                    winsound.Beep(1000, 500) 
                elif cls_id == 1:
                    print("💨 Smoke detected!")
                    winsound.Beep(600, 500) 

        cv2.imshow("🔥 Fire & Smoke Detection", annotated)

    except Exception as e:
        print("⚠️ Error fetching frame:", e)
        time.sleep(1)
        continue

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.05)

cv2.destroyAllWindows()
