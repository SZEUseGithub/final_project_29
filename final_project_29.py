import cv2
import numpy as np
import math;import random;import time
def gesture(defects,cnt,hull,crop_img):
    count_defects = 0;obtuse_angle=0
    if defects is None:
        count_defects = -2
    else:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0,0,255], -1)
            else:
                obtuse_angle += 1
            cv2.line(crop_img,start, end, [0,255,0], 2)
        if count_defects==0:
            if (cv2.contourArea(cnt)<0.9*cv2.contourArea(hull) and obtuse_angle>8):
                count_defects = 0
            else:
                count_defects = -1
        count_defects=count_defects+1
    return count_defects
def thresh(crop_img):
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh1
def draw2contours(thresh1,crop_img):
    contours, _ = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)
    return (contours,cnt,hull)
def printpoint(img,p1_point,p2_point):
    cv2.putText(img,"P1:"+str(p1_point),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(img,"P2:"+str(p2_point),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
def rand_rectangle(wid):
    row_lu0=random.randint(0,280);col_lu0=random.randint(20,120)
    row_rd0=row_lu0+wid;col_rd0=col_lu0+wid
    row_lu1=row_lu0;col_lu1=640-col_lu0-wid
    row_rd1=row_lu1+wid;col_rd1=640-col_lu0
    return (row_lu0,col_lu0,row_rd0,col_rd0,row_lu1,col_lu1,row_rd1,col_rd1)
def countdown(p1_point,p2_point,first_gesture,second_gesture):
    t_end=time.time()+2;cd=3
    while (time.time()<t_end):
        ret, img = cap.read()
        if time.time()>t_end-4/3:
            cd=2
        if time.time()>t_end-2/3:
            cd=1
        cv2.putText(img,"P1:"+str(p1_point),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(img,"P2:"+str(p2_point),(500,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(img,str(cd),(300,240),cv2.FONT_HERSHEY_SIMPLEX,5,(255,65,105),2,cv2.LINE_AA)
        cv2.putText(img,str(first_gesture),(300,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.putText(img,str(second_gesture),(300,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('Gesture', img)
        if cv2.waitKey(10) & 0xFF == 27:
            break



win0=0;win1=0;p1_point=0;p2_point=0;i=0;game=0
first_gesture=3;second_gesture=2;times=3;correct0=correct1=first_gesture;auto=1;
cap = cv2.VideoCapture(0)
for t in range(1,times+1):
    if auto==1:
        first_gesture=random.randint(2,5);second_gesture=random.randint(2,5)
        if first_gesture==second_gesture:
            second_gesture=first_gesture-1
    (row_lu0,col_lu0,row_rd0,col_rd0,row_lu1,col_lu1,row_rd1,col_rd1)=rand_rectangle(200)
    countdown(p1_point,p2_point,first_gesture,second_gesture)
    while(cap.isOpened()):
        # read image
        ret, img = cap.read()
        
        cv2.rectangle(img, (col_rd0,row_rd0), (col_lu0,row_lu0), (0,255,0),2,cv2.LINE_AA)
        cv2.rectangle(img, (col_rd1,row_rd1), (col_lu1,row_lu1), (0,255,0),2,cv2.LINE_AA)
        crop_img0 = img[row_lu0:row_rd0,col_lu0:col_rd0]
        crop_img1 = img[row_lu1:row_rd1,col_lu1:col_rd1]

        thresh1=thresh(crop_img1);thresh0=thresh(crop_img0);
        cv2.imshow('Thresholded1', thresh1);cv2.imshow('Thresholded0', thresh0);
        
        (contours0,cnt0,hull0)=draw2contours(thresh0,crop_img0);(contours1,cnt1,hull1)=draw2contours(thresh1,crop_img1)
        
        # drawing contours
        drawing0 = np.zeros(crop_img0.shape,np.uint8);drawing1 = np.zeros(crop_img1.shape,np.uint8)
        cv2.drawContours(drawing0, [cnt0], 0, (0, 255, 0), 0);cv2.drawContours(drawing1, [cnt1], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing0, [hull0], 0,(0, 0, 255), 0);cv2.drawContours(drawing1, [hull1], 0,(0, 0, 255), 0)
    
        # finding convex hull
        hull0_0 = cv2.convexHull(cnt0, returnPoints=False);hull1_0 = cv2.convexHull(cnt1, returnPoints=False)
    
        # finding convexity defects
        defects0 = cv2.convexityDefects(cnt0, hull0_0);defects1 = cv2.convexityDefects(cnt1, hull1_0)
        cv2.drawContours(thresh0, contours0, -1, (0, 255, 0), 3);cv2.drawContours(thresh1, contours1, -1, (0, 255, 0), 3)
    
        count_defects0=gesture(defects0,cnt0,hull0,crop_img0);count_defects1=gesture(defects1,cnt1,hull1,crop_img1)
        print(count_defects0,count_defects1)
    
        if count_defects0==correct0:
            win0=win0+1;correct0=second_gesture
        if count_defects1==correct1:
            win1=win1+1;correct1=second_gesture
    
        if win0==2:
            p1_point=p1_point+1
        if win1==2:
            p2_point=p2_point+1
        printpoint(img,p1_point,p2_point)
    
        # show appropriate images in windows
        all_img0 = np.hstack((drawing0, crop_img0));all_img1 = np.hstack((drawing1, crop_img1))
        cv2.imshow('Contours0', all_img0);cv2.imshow('Contours1', all_img1)
        print(game)
        
        if win0==2 or win1==2:
            correct0=correct1=first_gesture;win0=win1=0;game=game+1
            break
        cv2.imshow('Gesture', img)

        i=i+1
        # print(i)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    if game==times:
        if p1_point>p2_point:
            cv2.putText(img,"P1 is winner",(160,240),cv2.FONT_HERSHEY_SIMPLEX,2,(0,215,255),2,cv2.LINE_AA)
        else:
            cv2.putText(img,"P2 is winner",(160,240),cv2.FONT_HERSHEY_SIMPLEX,2,(0,215,255),2,cv2.LINE_AA)
        cv2.imshow('Gesture', img)
        break

cv2.waitKey();cv2.destroyAllWindows()
cap.release()