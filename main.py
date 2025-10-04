import cv2
import numpy as np
import json
from ultralytics import YOLO

#load up models 
ball_rim_model = YOLO("model/best_ball.pt")       # ball + rim
arc_model = YOLO("model/best_arc3.pt")            # arc segmentation
player_position_model = YOLO("yolov8n-pose.pt")   # player pose (ankles)

#load up video to track 
cap = cv2.VideoCapture("test_videos/al_dray_warriors.MP4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w, frame_h = int(cap.get(3)), int(cap.get(4))

# Output video (optional)
save_video = True
out = None
if save_video:
    out = cv2.VideoWriter("output_dray_fast.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (frame_w, frame_h))


#Tracking variables

fgm, fga = 0, 0
shot_in_progress = False
ball_was_above_rim = False
shot_result = None
shot_type = "Unknown"
shot_log = []

frame_count = 0
frame_skip = 3      # process every 3rd frame for speed
display_video = False  # toggle display window

print("Starting shot tracking...")


#Main loop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    #Skip frames for speed
    if frame_count % frame_skip != 0:
        continue

    current_time = frame_count / fps
    ball_center, rim_center, rim_box = None, None, None

    #detect ball and rim logic
    ball_rim_results = ball_rim_model.predict(frame, verbose=False)
    for r in ball_rim_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = ball_rim_model.names[cls]

            #Draw detections (ball & rim)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label.lower() == "ball":
                ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
            elif label.lower() == "rim":
                rim_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                rim_box = (x1, y1, x2, y2)


    #arc and player foot deteciton but not wokring well and taking extra time so leave out for now 
    # arc_mask_combined = np.zeros(frame.shape[:2], dtype='uint8')
    # arc_results = arc_model.predict(frame, verbose=False)
    # for r in arc_results:
    #     if r.masks is not None:
    #         for mask in r.masks.data.cpu().numpy():
    #             mask = (mask > 0.5).astype("uint8") * 255
    #             mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    #             arc_mask_combined = cv2.bitwise_or(arc_mask_combined, mask_resized)


    #foot detection for arc
    # foot_results = player_position_model.predict(frame, verbose=False)
    # shot_type = "Unknown"

    # if ball_center is not None:
    #     closest_player = None
    #     min_dist = float("inf")

    #     for r in foot_results:
    #         if r.keypoints is not None:
    #             for person in r.keypoints.xy.cpu().numpy():
    #                 left_ankle = tuple(map(int, person[15]))
    #                 right_ankle = tuple(map(int, person[16]))
    #                 hips = tuple(map(int, person[11]))
    #                 center_est = hips if hips else ((left_ankle[0] + right_ankle[0]) // 2,
    #                                                 (left_ankle[1] + right_ankle[1]) // 2)
    #                 dist = np.linalg.norm(np.array(center_est) - np.array(ball_center))
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     closest_player = (left_ankle, right_ankle)

    #     def is_inside_arc(mask, point, region=10, threshold=0.2):
    #         x, y = point
    #         h, w = mask.shape
    #         if not (0 <= x < w and 0 <= y < h):
    #             return False
    #         x1, y1 = max(0, x - region), max(0, y - region)
    #         x2, y2 = min(w, x + region), min(h, y + region)
    #         roi = mask[y1:y2, x1:x2]
    #         return np.mean(roi > 0) > threshold

    #     if closest_player:
    #         left_ankle, right_ankle = closest_player
    #         if np.sum(arc_mask_combined) == 0:
    #             shot_type = "2PT"
    #         else:
    #             inside_left = is_inside_arc(arc_mask_combined, left_ankle)
    #             inside_right = is_inside_arc(arc_mask_combined, right_ankle)
    #             shot_type = "2PT" if (inside_left or inside_right) else "3PT"


    #detect the ball and shot and calc 
    if ball_center and rim_center:
        #detect new shot
        if not shot_in_progress and ball_center[1] < rim_center[1]:
            shot_in_progress = True
            ball_was_above_rim = True

        #ball goes below rim again then shot ends
        elif shot_in_progress and ball_center[1] > rim_center[1]:
            fga += 1
            if abs(ball_center[0] - rim_center[0]) < 50:
                fgm += 1
                shot_result = "MAKE"
                color = (0, 255, 0)
            else:
                shot_result = "MISS"
                color = (0, 0, 255)

            shot_log.append({
                "time": round(current_time, 2),
                "result": shot_result,
                "shot_type": shot_type
            })

            shot_in_progress = False
            ball_was_above_rim = False

            cv2.putText(frame, shot_result, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    #Overlay stats

    cv2.putText(frame, f"FGM/FGA: {fgm}/{fga}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    #Output

    if save_video:
        out.write(frame)

    if display_video:
        cv2.imshow("Shot Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

#save log of shots made and attempts
with open("shot_log.json", "w") as f:
    json.dump(shot_log, f, indent=4)
print(f"Done. Logged {len(shot_log)}")