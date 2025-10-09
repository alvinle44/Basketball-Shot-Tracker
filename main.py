import cv2
import math
import json
import numpy as np
import torch
from ultralytics import YOLO
from utils import get_device, smooth_point, detect_up, detect_down, score_prediction

#load the device 
device = get_device()
#load trained ball and rim tracking model 
model = YOLO("model/best_ball.pt")

#load video to be processed 
cap = cv2.VideoCapture("test_videos/al_dray_warriors.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))

#location to write labeled video to 
out = cv2.VideoWriter("outputs/output_geo_merge.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


#keep track of ball ids
#missing frames keeps track of how long a ball disappears for and handles cases where ball is reassigned a new id
#last_state keeps track of state of ball being attempted, made, missed
balls, missing_frames, last_state = {}, {}, {}

#velocity keeps track of the speed of the ball being shot
#cooldowns prevents balls from being count as a double make or miss
velocity, cooldowns = {}, {}
rim_box, rim_center = None, None
fgm, fga = 0, 0
frame_idx = 0


COOLDOWN_FRAMES = int(fps * 0.6)
MAX_DISTANCE = 100
MAX_MISSING_FRAMES = 15
CONF_THRESHOLD = 0.35
NET_DEPTH = 120


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model.predict(frame, verbose=False, device=device, conf=CONF_THRESHOLD)
    detections = []


    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls].lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2)//2, (y1 + y2)//2)

            if "rim" in label or "hoop" in label:
                rim_box, rim_center = (x1, y1, x2, y2), center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            elif "ball" in label and conf > CONF_THRESHOLD:
                detections.append(center)
                cv2.circle(frame, center, 4, (0,0,255), -1)

    if not rim_box:
        out.write(frame)
        continue

    rx1, ry1, rx2, ry2 = rim_box
    assigned = set()

    for center in detections:
        cx, cy = center
        matched_id, min_dist = None, float("inf")

        for bid, traj in balls.items():
            px, py = traj[-1]
            dist = math.hypot(px - cx, py - cy)
            if dist < MAX_DISTANCE and dist < min_dist:
                matched_id, min_dist = bid, dist

        if matched_id is None:
            new_id = max(balls.keys(), default=-1) + 1
            balls[new_id] = [center]
            missing_frames[new_id] = 0
            last_state[new_id] = "init"
            velocity[new_id] = (0, 0)
            assigned.add(new_id)

            to_merge = None
            for old_id, missed in list(missing_frames.items()):
                if old_id == new_id:
                    continue
                if 0 < missed <= 8:  # short dropout
                    old_traj = balls.get(old_id, [])
                    if not old_traj:
                        continue
                    ox, oy = old_traj[-1]
                    cx, cy = center
                    dist = math.hypot(cx - ox, cy - oy)

                    if rim_box:
                        rim_center_y = (ry1 + ry2) / 2
                        old_in_rim = (ry1 - 10) <= oy <= (ry2 + 40)
                        new_below_rim = cy >= rim_center_y
                        vertical_ok = old_in_rim and new_below_rim
                        near_rim_zone = (rx1 - 150 < cx < rx2 + 150 and
                                         ry1 - 180 < cy < ry2 + 180)
                    else:
                        near_rim_zone = False
                        vertical_ok = False

                    if dist < 130 and near_rim_zone and vertical_ok:
                        print(f"🔁 Merging old ID {old_id} → new ID {new_id} (below rim handoff)")
                        merged_traj = old_traj + [center]
                        balls[new_id] = merged_traj[-50:]
                        velocity[new_id] = velocity.get(old_id, (0, 0))
                        last_state[new_id] = last_state.get(old_id, "init")
                        cooldowns[new_id] = cooldowns.get(old_id, 0)
                        missing_frames[new_id] = 0
                        assigned.add(new_id)

                        cv2.line(frame, (int(ox), int(oy)), (int(cx), int(cy)), (0,255,255), 3)
                        to_merge = old_id
                        break

            if to_merge is not None:
                for d in (balls, missing_frames, last_state, velocity, cooldowns):
                    d.pop(to_merge, None)

        else:
            smoothed = smooth_point(balls[matched_id][-1], center)
            balls[matched_id].append(smoothed)
            if len(balls[matched_id]) > 50:
                balls[matched_id].pop(0)
            missing_frames[matched_id] = 0
            assigned.add(matched_id)


    for bid in list(balls.keys()):
        traj = balls[bid]

        if bid not in assigned:
            missing_frames[bid] += 1
            bx, by = traj[-1]
            near_rim = (rx1 - 150 < bx < rx2 + 150 and
                        ry1 - 150 < by < ry2 + 150)

            # Predict forward a few frames to help reconnect
            if len(traj) >= 3 and missing_frames[bid] <= 5:
                (x1, y1), (x2, y2) = traj[-2], traj[-1]
                vx, vy = x2 - x1, y2 - y1
                predicted = (x2 + vx, y2 + vy)
                balls[bid].append(predicted)

            if missing_frames[bid] > (25 if near_rim else 10):
                for d in (balls, missing_frames, last_state, velocity, cooldowns):
                    d.pop(bid, None)
                continue

        if len(traj) >= 2:
            vx, vy = traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1]
            velocity[bid] = (vx, vy)

        # --- Geometric shot detection ---
        if len(traj) < 5:
            continue

        bx, by = traj[-1]
        vy = velocity[bid][1]
        old_state = last_state.get(bid, "init")

        if detect_up(traj, rim_box) and vy > 0:
            if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                fga += 1
                cooldowns[bid] = frame_idx
                last_state[bid] = "attempting"
                print(f"🎯 Shot attempt for Ball {bid} at frame {frame_idx}")

        if last_state.get(bid) in ["attempting", "init"]:
            if detect_down(traj, rim_box) and score_prediction(traj, rim_box):
                if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                    fgm += 1
                    cooldowns[bid] = frame_idx
                    last_state[bid] = "made"
                    continue
            elif detect_down(traj, rim_box) and not score_prediction(traj, rim_box):
                if bid not in cooldowns or frame_idx - cooldowns[bid] > COOLDOWN_FRAMES:
                    cooldowns[bid] = frame_idx
                    last_state[bid] = "missed"
                    continue

        if by > h - 40:
            for d in (balls, missing_frames, last_state, velocity, cooldowns):
                d.pop(bid, None)
            continue

        for k in range(1, len(traj)):
            pt1 = (int(traj[k-1][0]), int(traj[k-1][1]))
            pt2 = (int(traj[k][0]), int(traj[k][1]))
            cv2.line(frame, pt1, pt2, (0,0,255), 2)

        cx, cy = traj[-1]
        cv2.putText(frame, f"ID {bid}", (int(cx)+10, int(cy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.putText(frame, f"FGM/FGA: {fgm}/{fga}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    out.write(frame)
    cv2.imshow("Shot Tracker (Geo + Merge)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- END -------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done. Logged {fgm} / {fga}")
with open("shot_log.json", "w") as f:
    json.dump({"FGM": fgm, "FGA": fga}, f, indent=4)
