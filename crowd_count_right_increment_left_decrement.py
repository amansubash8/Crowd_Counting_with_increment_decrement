import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === CONFIGURATION ===
MODEL_PATH = "/Users/amansubash/Downloads/crowd_counting_weights.pt"
VIDEO_PATH = "/Users/amansubash/Downloads/l.mp4"
OUTPUT_PATH = "/Users/amansubash/Downloads/a.mp4"
CONFIDENCE_THRESHOLD = 0.7
FRAME_SKIP = 1
ROI_TOP_LEFT = (620, 0)
ROI_BOTTOM_RIGHT = (820, 500)

# === INITIALIZE ===
model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=20)
track_memory = {}  # track_id -> { 'positions': [(x, y, frame)], 'counted': False, 'inside_roi': bool }
total_inside = 100

def is_inside_roi(x, y):
    """Check if point (x, y) is inside the ROI"""
    return (ROI_TOP_LEFT[0] <= x <= ROI_BOTTOM_RIGHT[0] and 
            ROI_TOP_LEFT[1] <= y <= ROI_BOTTOM_RIGHT[1])

def run_people_counter():
    global total_inside

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP == 0:
            detections = []

            # Run detection on ROI
            roi_frame = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
            results = model(roi_frame, conf=CONFIDENCE_THRESHOLD)[0]

            for r in results.boxes.data:
                x1, y1, x2, y2, conf, cls = r
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1_full = x1 + ROI_TOP_LEFT[0]
                y1_full = y1 + ROI_TOP_LEFT[1]
                x2_full = x2 + ROI_TOP_LEFT[0]
                y2_full = y2 + ROI_TOP_LEFT[1]
                w, h = x2_full - x1_full, y2_full - y1_full
                detections.append(([x1_full, y1_full, w, h], float(conf)))

            tracks = tracker.update_tracks(detections, frame=frame)

            # Process tracks
            active_track_ids = set()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                active_track_ids.add(track_id)
                
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Initialize track memory if new
                if track_id not in track_memory:
                    track_memory[track_id] = {
                        "positions": [],
                        "counted": False,
                        "inside_roi": False,
                        "last_position": None
                    }

                # Check if person is currently inside ROI
                currently_inside = is_inside_roi(center_x, center_y)
                
                # Store position history (keep last 10 positions)
                track_memory[track_id]["positions"].append((center_x, center_y, frame_id))
                if len(track_memory[track_id]["positions"]) > 10:
                    track_memory[track_id]["positions"].pop(0)

                # Track ROI state changes
                was_inside = track_memory[track_id]["inside_roi"]
                track_memory[track_id]["inside_roi"] = currently_inside

                # Exit detection logic
                if was_inside and not currently_inside and not track_memory[track_id]["counted"]:
                    # Person has exited the ROI
                    last_pos = track_memory[track_id]["last_position"]
                    if last_pos is not None:
                        last_x, last_y = last_pos
                        
                        # Determine exit direction based on current position relative to ROI
                        roi_center_x = (ROI_TOP_LEFT[0] + ROI_BOTTOM_RIGHT[0]) // 2
                        
                        if center_x > ROI_BOTTOM_RIGHT[0]:  # Exited to the right
                            total_inside += 1
                            print(f"🟢 EXIT RIGHT | ID {track_id} | +1 | Total: {total_inside}")
                            track_memory[track_id]["counted"] = True
                            
                        elif center_x < ROI_TOP_LEFT[0]:  # Exited to the left
                            total_inside = max(0, total_inside - 1)
                            print(f"🔻 EXIT LEFT | ID {track_id} | -1 | Total: {total_inside}")
                            track_memory[track_id]["counted"] = True

                # Update last position
                track_memory[track_id]["last_position"] = (center_x, center_y)

                # Draw bounding box and ID
                box_color = (0, 255, 0) if currently_inside else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 3, box_color, -1)

            # Clean up old tracks that are no longer active
            inactive_tracks = set(track_memory.keys()) - active_track_ids
            for track_id in list(inactive_tracks):
                # Check if track exited before being removed
                if track_id in track_memory and track_memory[track_id]["inside_roi"]:
                    if track_memory[track_id]["last_position"] and not track_memory[track_id]["counted"]:
                        last_x, last_y = track_memory[track_id]["last_position"]
                        
                        # Estimate exit direction based on trajectory
                        positions = track_memory[track_id]["positions"]
                        if len(positions) >= 2:
                            # Calculate movement direction
                            start_x = positions[0][0]
                            end_x = positions[-1][0]
                            
                            if end_x > start_x:  # Moving right
                                total_inside += 1
                                print(f"🟢 EXIT RIGHT (Lost) | ID {track_id} | +1 | Total: {total_inside}")
                            else:  # Moving left
                                total_inside = max(0, total_inside - 1)
                                print(f"🔻 EXIT LEFT (Lost) | ID {track_id} | -1 | Total: {total_inside}")
                
                # Remove old track from memory
                del track_memory[track_id]

        # Draw ROI box
        cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 0, 255), 3)
        
        # Draw exit zones
        cv2.line(frame, (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1]), (ROI_TOP_LEFT[0], ROI_BOTTOM_RIGHT[1]), (255, 0, 0), 3)  # Left exit
        cv2.line(frame, (ROI_BOTTOM_RIGHT[0], ROI_TOP_LEFT[1]), (ROI_BOTTOM_RIGHT[0], ROI_BOTTOM_RIGHT[1]), (0, 255, 0), 3)  # Right exit

        # Overlay count
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"People Count: {total_inside}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Active Tracks: {len(active_track_ids)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Exit-Based People Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n📊 Final Count: {total_inside}")
    print(f"💾 Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_people_counter()