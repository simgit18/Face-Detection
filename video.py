import cv2, time, bokeh, pandas
from datetime import datetime

df=pandas.DataFrame(columns=["Start","End"])

status_list=[None,None]
times=[]
face_cascade = cv2.CascadeClassifier("cascade.xml")

first_frame= None
 
video = cv2.VideoCapture(0,cv2.CAP_DSHOW)


while True:

    check, frame= video.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),0)
    status=0

    if first_frame is None:
        first_frame= gray
        continue
    
    
    delta_frame= cv2.absdiff(first_frame, gray)
    thres_frame = cv2.threshold( delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
    thres_frame = cv2.dilate(thres_frame, None, iterations=2)
    

    (cest,_) = cv2.findContours(thres_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cest:
        if cv2.contourArea(contour) < 10000:
            continue
        
        
        status=1
        (x,y,z,w) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y),(x+z,y+w),(255,0,0),3)

    status_list.append(status)

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    cv2.imshow("capture",frame)
    cv2.imshow("delta", delta_frame)
    cv2.imshow("thresh", thres_frame)
    
    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
    # print(status)
video.release()


for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

cv2.destroyAllWindows()
print(times)

print(df)



df.to_csv("date_and_time.csv")