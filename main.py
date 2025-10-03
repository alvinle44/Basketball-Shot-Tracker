import cv2 
from ultralytics import YOLO
import numpy as np

# Load models for ball detection and arc detecion 
#need to add player foot detection next 
ball_rim_model = YOLO("model/best_ball.pt")
arc_model = YOLO("model/best_arc2.pt")

#open up video 
cap = cv2.VideoCapture("test_videos/alvin_uci_shot.MP4")

#next implement the tracking portion
fgm, fga = 0, 0

three_ptm, three_pta = 0,0

while True:
    #read from from video
    ret, frame = cap.read()
    if not ret:
        break

    #detect the ball and rim and if detected, track the center of the ball 
    #need person with ball detction of feet as well so imp that next 
    ball_center = None
    ball_rim_results = ball_rim_model.predict(frame, verbose=False)

    #iterate through each detected ball and rim found in frame
    for r in ball_rim_results:
        for box in r.boxes:

            #get coords for box and label class and confidence score of box 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = ball_rim_model.names[cls]

            #draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            #if ball, get the center value of ball to compute next if going towards rim and suhc to track makes or misses 
            if label.lower() == "ball":
                ball_center = ((x1 + x2)//2, (y1 + y2)//2)
                #red dot for ball center
                cv2.circle(frame, ball_center, 5, (0,0,255), -1)  

    #in currrent frame, also check where the ball is by using the arc tracker to determine if a 2 or three point shot. 
    shot_type = "Unknown"
    arc_results = arc_model.predict(frame, verbose=False)

    #iter through each results for arc 
    for r in arc_results:
        #if there is masks in the results that means the arc is detected
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            #iter through masks scores that are confident in arc 
            for mask in masks:
                #convert float mask to binary mask and resize to fit frame of video 
                mask = (mask > 0.5).astype("uint8") * 255
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Make transparent overlay of mask to show the arc line 
                color_mask = cv2.merge([
                    np.zeros_like(mask_resized),
                    np.zeros_like(mask_resized),
                    mask_resized
                ])
                frame = cv2.addWeighted(frame, 1, color_mask, 0.4, 0)

                #check if ball is detected and if so where is it located 
                #im not sure what is more important i think it should be user with ball that needs to be tracked next to determine if its a shot or not
                if ball_center is not None:
                    if mask_resized[ball_center[1], ball_center[0]] > 0:
                        shot_type = "2PT"
                    else:
                        shot_type = "3PT"

    #if the ball is found display the shot andb all center with text about shot type if the ball were to be shot 
    if ball_center is not None:
        cv2.putText(frame, shot_type, (ball_center[0]+10, ball_center[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

    #display frame
    cv2.imshow("Ball, Rim & Arc Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()