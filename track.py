import cv2
import numpy as np

# define range of green color in HSV
blue_lower=np.array([45, 100, 50],np.uint8)
blue_upper=np.array([75,255,255],np.uint8)

# Initalize camera
cap=cv2.VideoCapture(0)

# Create empty points array
points=[]

while(1):   
    # Capture webcame frame     
    ret,frame=cap.read() 
    Height, Width = frame.shape[:2]
    
    # Threshold the HSV image to get only green colors
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #masking
    mask=cv2.inRange(hsv,blue_lower,blue_upper)
    
    #applying dilation and erode
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #finding contours
    _,contours,_=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center =None
    if len(contours)>0:
        # Get the largest contour and its center 
        c=max(contours,key=cv2.contourArea)
        (x,y),radius=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except:
            center =None
         # Allow only countors that have a larger than 20 pixel radius
        if radius>20:
            
         #creating circle to the contour   
         cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
         cv2.circle(frame, center, 5, (0, 255, 0), -1)
    
    #appending centers 
    points.append(center)

    for i in range(1,len(points)):
     if points[i - 1] is None or points[i] is None:
      continue
     cv2.line(frame,points[i-1],points[i],(0,255,255),5)
    #flipping the frame 
    frame = cv2.flip(frame, 1)
    
    #displaying
    cv2.namedWindow("kk",cv2.WINDOW_NORMAL)
    cv2.imshow("kk", frame)    
     
    if cv2.waitKey(1) ==ord('q'): #q to close the frame
       break

# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()     
        
