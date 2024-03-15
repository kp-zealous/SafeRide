from ultralytics import YOLO
import cv2
import csv
import cvlib as cv
from cvlib.object_detection import draw_bbox
from time import gmtime, strftime
#import final_number_plate as f
import time
import numpy as np
import math
import cv2
import imagecheck


def store(data):
     with open(r'E:\kp\TRANSPORT PROJ DAY 2\classification\vehicledata.csv', 'w', newline='') as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Check if the file is empty
        if csvfile.tell() == 0:
           writer.writeheader()  # Write the header if it's the first entry
        writer.writerow(data)
def setFps(PTime):
    CTime = time.time()
    fp = 1/(CTime - PTime)
    PTime = CTime
    return(fp,PTime)
   

def setEstimate(PTime,x, y):
    d_pixels = math.sqrt(x**2 + y**2)
    fp,new_PTime=setFps(PTime)
    ppm = 8.8
    d_meters = int(d_pixels/ppm)
    speed = d_meters/(fp*3.6)
    speed_in_km =np.average(speed)
    return (speed_in_km/1000000000)        
def classify(videopath):
    d={'date':None,'TOTAL':None}
    count=0
    ww=80
    hh=80
    offset=2
    y1=615
    carros=0
    k=1
    PTime=0
    i=1
    previous_image = None
    detect=[]
    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output=r"E:\kp\TRANSPORT PROJ DAY 2\outputfile2.mp4"
    video_writer= cv2.VideoWriter(output, fourcc, fps, (800, 1200))
  
    model = YOLO(r"E:\kp\TRANSPORT PROJ DAY 2\classification\yolov8n.pt")

    while cap.isOpened():
        ret, frame = cap.read()
        t=10
        m=1
        if not ret:
            break
        frame=cv2.resize(frame,(800,1200))
        results = model.predict(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.6, model=model)
        cv2.line(frame,(0,y1),(800,y1),(0,127,255),2)
        for (x, y, w, h) in bbox:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            gray_plates = gray[y:y+h, x:x+w]
            color_plates = frame[y:y+h, x:x+w]
            if previous_image is not None:
                if not imagecheck.are_images_same(previous_image, gray_plates):
                    cv2.imwrite(r'E:\kp\TRANSPORT PROJ DAY 2\npd\output\Numberplate{}.jpg'.format(i), gray_plates)
                    cv2.imshow('Number Plate', gray_plates)
            #cv2.imshow('Number Plate Image', frame)
        i+=1
        previous_image=gray_plates
        
        
        for l, c ,b in zip(label, conf,bbox):
            x,y,w,h=b
            center=(int((x+w)/2), int((y+h)/2))
            detect.append(center)
            g=setEstimate(PTime,x, y)
            cv2.circle(frame, center, 2, (0,0,255), 2)
            cv2.putText(frame,str(int(g))+"Km/h",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255, 255, 0),2)
            showtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            d['date']=showtime
            d['TOTAL']=count
            #print(f"Detected object: {l} with confidence level of {c}n")
            for (x,y) in detect:
                if (y<=(y1+offset)) and (y>=(y1-offset)):
                    carros+=1
                    detect.remove((x,y))
                    #gh=f.npd(frame,k)
                    print("cars detected:"+str(carros))
           
                    if l not in d:
                       d[l]=1
                       count+=1
               
                    else:
                       d[l]+=1
                       count+=1
        store(d)
        output_image = draw_bbox(frame, bbox, label, conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        n=len(d)
        for i,j in zip(d.keys(),d.values()):
            if m<=n:
                cv2.putText(frame, str(i)+':'+str(j), (5,(20+m*t)), font, 0.75, (0, 255, 255), 2, cv2.LINE_4)
                t+=20
        #cv2.putText(frame, str(carros), (5,40), font, 0.75, (0, 255, 255), 2, cv2.LINE_4)
        cv2.imshow("vehicles",frame)
        video_writer.write(frame)
        k+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Release the VideoWriter object
    video_writer.release()

videopath=r"C:\Users\TRANING-9\Downloads\whatsapp-video-2023-06-29-at-94348-am_pi7y5Nlk.mp4"
classify(videopath)
#C:\Users\TRANING-9\Downloads\clear.mp4

