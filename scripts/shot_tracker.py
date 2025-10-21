import cv2
import math
import json
import numpy as np
import torch
from ultralytics import YOLO
from scripts.utils import get_device, smooth_point, detect_up, detect_down, score_prediction
from pathlib import Path

def process_video(video_path=None, output_path=None, return_video=False):
    #process video
    #load the device 
    device = get_device()
    #load trained ball and rim tracking model 
    model = YOLO("model/best_ball.pt")

    #load video to be processed 
    if not video_path:
        cap = cv2.VideoCapture(0)
    elif video_path:
        cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))

    #location to write labeled video to 
    if return_video:
        Path("outputs").mkdir(exist_ok=True)
        out = cv2.VideoWriter(output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    #keep track of ball ids
    #missing frames keeps track of how long a ball disappears for and handles cases where ball is reassigned a new id
    #last_state keeps track of state of ball being attempted, made, missed
    #balls dictionary key:ball id, values: [(x,y)]
    #missing_frames keeps track of how many frames a ball disappeared key:ball_id value: counter for how many times for each frame the ball id is not present in a row before being detected again
    #last_state keeps track of key:ball ID, value: the ball state as in init, attempted, made, missed
    balls, missing_frames, last_state = {}, {}, {}

    #velocity keeps track of the speed of the ball being shot
    #cooldowns prevents balls from being count as a double make or miss
    #velcity key: ball id and values (vx, vy) the velcoity spped for each ball 
    #cooldowns prevents the ball from being counted by a double shot in the cooldown window, key:ball id value: frame the ball was last counted as make/miss attempt
    velocity, cooldowns = {}, {}

    #at each frame make sure the rim is in frame by calculating the rim_box and rim_center if not the bypass current frame because you 
    #cannot check if a shot is made or missed if the rim is out of frame
    rim_box, rim_center = None, None

    #counters for shot attemps and made through whole duration
    fgm, fga = 0, 0
    #count for how many frames have passed by 
    frame_idx = 0

    #cooldown for ball between shot attempts
    COOLDOWN_FRAMES = int(fps * 0.6)
    #use this to match the ball when tracker loses ball in between frames
    MAX_DISTANCE = 100
    #if ball is missing for this many frames, remove it from all data structures
    MAX_MISSING_FRAMES = 15
    #yolo detection threshold
    CONF_THRESHOLD = 0.35

    #begin analyzing frame by frame
    #later on skip frames to increase speed of inference 
    while True:
        #read single frame from the video
        ret, frame = cap.read()
        #break out of loop is no more frames
        if not ret:
            break
        #increment frame count 
        frame_idx += 1

        #run yolo model on the current frame 
        #convert to streaming later on to make the logic much more effiecient as it does not store frame by frame
        #streaming does not store frame by frame but produces outputs as it is being read/ran
        results = model.predict(frame, verbose=False, device=device, conf=CONF_THRESHOLD)

        #stores the centers of the balls detected in the current frame
        detections = []

        #iterate through all of the objects detected in the current frame 
        for r in results:
            #iterate through each detected box 
            for box in r.boxes:
                cls = int(box.cls[0]) #get class id of the object 0 = ball 1 = rim
                label = r.names[cls].lower() #get class name corr to the cls of the object detected
                conf = float(box.conf[0]) #get the detected object confidnece score outputted by the model 
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding box coords
                center = ((x1 + x2)//2, (y1 + y2)//2) #center position of the box 

                if "rim" in label:
                    #create a rectangle around the rim to display that the rim is detected 
                    
                    rim_box, rim_center = (x1, y1, x2, y2), center
                    if return_video:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                elif "ball" in label and conf > CONF_THRESHOLD:
                    #mark center of ball with dot and add to detection array so the coords of the ball can be traced later
                    detections.append(center)
                    if return_video:
                        cv2.circle(frame, center, 4, (0,0,255), -1)

        #if no rim was detectd, this frame can be skipped 
        #later on implement logic for two rims and whichever rim the ball is closest to assign to that rim for make/miss tracking for now it is ok 
        if not rim_box:
            out.write(frame)
            continue
        #get coords of the rim detected
        rx1, ry1, rx2, ry2 = rim_box
        assigned = set() #keeps track of the ball ids found in curr frame and used to compare all of the active ball ids. if the ball is gone for too long, then remove from the ball id tracker

        #iter through all detected balls 
        for center in detections:
            #find the ball with the last nearest postion to the found center, i did this because often times the ball would become distored and out of frame, so it would lose track of the ball and reassing to new id 
            cx, cy = center
            matched_id, min_dist = None, float("inf")

            #iter throgh all balls and find the one that was the closest to this current center position and if close enough then assign that center to previos balls
            for bid, traj in balls.items():
                #get last position of the ball
                px, py = traj[-1]
                #calc distance of the current center to the previous center of current ball
                dist = math.hypot(px - cx, py - cy)
                #if it is the min distance to ball out of all balls and is within the distance, assign ball to that id 
                if dist < MAX_DISTANCE and dist < min_dist:
                    matched_id, min_dist = bid, dist

            #if this is a new ball or ball has been out of frame for too long, create new ball id 
            if matched_id is None:
                #assign new ball id 
                new_id = max(balls.keys(), default=-1) + 1
                balls[new_id] = [center] #assign to new ball id the corrds of ball center
                missing_frames[new_id] = 0 #assign to the missing_frames dict for new ball id to 0 
                last_state[new_id] = "init" #assign the ball_id state to inital state
                velocity[new_id] = (0, 0) #since new ball assign to ball_id 0 velocity 
                assigned.add(new_id) #add to assigned the ball in curr frame

                #check once aggain if the ball reappeard or dropped out just in case it skipped the ball for a few frames
                #important because it minimizes the number of attempts because if ball is lost mid air it will count as a shot attempt
                to_merge = None #hold the id of the old ball that will be merged if found match 

                #begin looping through each ball and see how many frames it has been missing for 
                #if ball has been missing for short period only then there is a chance we can reassign this new ball to this old ball id 
                for old_id, missed in list(missing_frames.items()):
                    if old_id == new_id:
                        continue
                    #if the ball is only missing for a few frames then we can begin to check if this new ball is actually an old ball
                    if 0 < missed <= 8:  #short dropout
                        #if potential old ball, get the path/trajectory of the ball 
                        old_traj = balls.get(old_id, [])
                        #skip of no ball history to compare to 
                        if not old_traj:
                            continue

                        #get coords of ball last known position 
                        ox, oy = old_traj[-1]
                        #coords to the new ball center in this frame
                        cx, cy = center
                        #calc the distance between the two 
                        dist = math.hypot(cx - ox, cy - oy)

                        """
                        This logic here helps me detect if the ball just went through the net or not
                        My tracker has trouble tracing the ball as it goes through the net, so this helps me determine if this ball just fell through
                        the net and reappeared or not.
                        """
                        if rim_box:
                            rim_center_y = (ry1 + ry2) / 2 #get rim center
                            old_in_rim = (ry1 - 10) <= oy <= (ry2 + 40) #get position of the old balls last known position to check if the ball is near or around rim 
                            #checks if ball is between 10 pixels above and 40 pixels below the rim before it vanished 
                            new_below_rim = cy >= rim_center_y #if the ball is now below the rim we can now be certain the ball either missed or went through the net 
                            #if oldball was in rim and new ball is below the rim we can be confident that the ball just went through the hoop 
                            vertical_ok = old_in_rim and new_below_rim
                            #checks if the ball is within the region near the hoop 
                            near_rim_zone = (rx1 - 150 < cx < rx2 + 150 and
                                            ry1 - 180 < cy < ry2 + 180)
                        else:
                            near_rim_zone = False
                            vertical_ok = False
                        
                        #if ball is close enough to the rim, we can merge the ball ids to one 
                        if dist < 130 and near_rim_zone and vertical_ok:
                            #merges new and old ball trajectory 
                            merged_traj = old_traj + [center]
                            balls[new_id] = merged_traj[-50:]

                            #copies old tracking info into new id 
                            velocity[new_id] = velocity.get(old_id, (0, 0))
                            last_state[new_id] = last_state.get(old_id, "init")
                            cooldowns[new_id] = cooldowns.get(old_id, 0)
                            #reset missing frame to 0 because ball was jsut found 
                            missing_frames[new_id] = 0
                            #add to ball found set 
                            assigned.add(new_id)
                            
                            #stores the old ball id so we can delete it later because the old ball was found but assigned a new id 
                            to_merge = old_id
                            break
                
                #if the ball was found and merged remove the ball id from all data strucutres no longer need to keep track 
                if to_merge is not None:
                    for d in (balls, missing_frames, last_state, velocity, cooldowns):
                        d.pop(to_merge, None)

            else:
                #if the ball in the frame already had a found id, dont need logic for merge detection 
                #smoothed gets the ball id and current center in the case teh ball jitteres 
                #produces average based on the last position to smooth jumps 
                smoothed = smooth_point(balls[matched_id][-1], center)
                balls[matched_id].append(smoothed)

                #dont need too many points on ball to remove old postions of the ball if there is a new one 
                if len(balls[matched_id]) > 50:
                    balls[matched_id].pop(0)
                #assign to ball that it is not missing frame by updating frames missed to 0 
                missing_frames[matched_id] = 0
                assigned.add(matched_id)

        
        #go through each ball that is currently being tracked 
        for bid in list(balls.keys()):
            #get ball trajectory/path 
            traj = balls[bid]
            
            #if the ball is in the ball id tracker, but not in this current frame, increment its mising frame by 1 
            if bid not in assigned:
                missing_frames[bid] += 1

                #check once again if the ball pos was near the rim area 
                bx, by = traj[-1]
                near_rim = (rx1 - 150 < bx < rx2 + 150 and
                            ry1 - 150 < by < ry2 + 150)

                # Predict forward a few frames to help reconnect, if the ball is missing, try to preduct wher the ball would be currently just in case the ball does reappear, we extrapolate another point for the ball 
                if len(traj) >= 3 and missing_frames[bid] <= 5:
                    (x1, y1), (x2, y2) = traj[-2], traj[-1]
                    vx, vy = x2 - x1, y2 - y1
                    predicted = (x2 + vx, y2 + vy)
                    balls[bid].append(predicted)

                #if ball is missing from memory for too long, remove from memory 
                if missing_frames[bid] > (25 if near_rim else 10):
                    for d in (balls, missing_frames, last_state, velocity, cooldowns):
                        d.pop(bid, None)
                    continue
                
            #if the ball is found, update the velocity of the ball and add to the velocity tracker 
            if len(traj) >= 2:
                vx, vy = traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1]
                velocity[bid] = (vx, vy)

            #if the ball has not been in the air for enough frames, dont need to predict shot make or miss yet
            if len(traj) < 5:
                continue
            
            #get ball coords
            bx, by = traj[-1]
            vy = velocity[bid][1] #ball vertical speed 
            old_state = last_state.get(bid, "init") #get the current state of ball 


            #if the ball is going up/being attemtped
            if detect_up(traj, rim_box) and vy > 0:
                #if the ball is being attemtped and not in the cooldown zone, increment the shot attempt by 1 as the ball goes up
                #this prevents the ball from being count as two shots 
                if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                    fga += 1
                    #update the frame the ball is being considered a shot as
                    cooldowns[bid] = frame_idx
                    #update the state of the ball 
                    last_state[bid] = "attempting"
            

            #eval balls that are in a state of being shot 
            if last_state.get(bid) in ["attempting", "init"]:
                #check if the ball is falling below the rim and predict if the ball went in or not 
                if detect_down(traj, rim_box) and score_prediction(traj, rim_box):
                    #if both are true, we predict the shot as made
                    if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                        #increment the shot made
                        fgm += 1
                        #set cool down as current frame to ensure the ball is not double counted 
                        cooldowns[bid] = frame_idx
                        last_state[bid] = "made"
                        continue
                #checks if ball below rim and if predicts make or miss
                elif detect_down(traj, rim_box) and not score_prediction(traj, rim_box):
                    if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                        #set state as miss and set the cooldown frame to ensure ball has a little bit of time before being count as a shot again
                        cooldowns[bid] = frame_idx
                        last_state[bid] = "missed"
                        continue
            #remove balls that are no longer in frame to keep memory clean 
            if by > h - 40:
                for d in (balls, missing_frames, last_state, velocity, cooldowns):
                    d.pop(bid, None)
                continue
            #draw trajectory of the ball, path that connects teh ball postions to see the ball path 
            if return_video:
                for k in range(1, len(traj)):
                    pt1 = (int(traj[k-1][0]), int(traj[k-1][1]))
                    pt2 = (int(traj[k][0]), int(traj[k][1]))
                    cv2.line(frame, pt1, pt2, (0,0,255), 2)
                #label the ball id on the video 
                cx, cy = traj[-1]
                cv2.putText(frame, f"ID {bid}", (int(cx)+10, int(cy)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        #display the text on screen of field goal make/attempt count 
        if return_video:
            cv2.putText(frame, f"FGM/FGA: {fgm}/{fga}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        if return_video:
            out.write(frame)
        # cv2.imshow("Shot Tracker (Geo + Merge)", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    if out is not None:
        out.release()
    if return_video:
        cv2.destroyAllWindows()
    print(f"Done. Logged {fgm} / {fga}")
    with open("shot_log.json", "w") as f:
        json.dump({"FGM": fgm, "FGA": fga}, f, indent=4)
    return {"FGM": fgm, "FGA": fga}