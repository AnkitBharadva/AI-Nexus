import cv2
import time
import numpy as np
from ultralytics import YOLO

# --- Configuration & Hyperparameters ---
VIDEO_SOURCES = [
    r"D:\tm\12207144_1920_1080_30fps.mp4",
    r"D:\tm\Cars Moving On Road Stock Footage - Free Download_1080p.mp4",
    r"D:\tm\istockphoto-866517852-640_adpp_is.mp4",
    r"D:\tm\12207144_1920_1080_30fps.mp4" # Reusing video for test
]

BASE_GREEN_TIME = 10.0              # Minimum baseline green light (seconds)
MAX_GREEN_TIME = 120.0              # Maximum allowed green time (seconds)
TIME_PER_VEHICLE = 0.7              # Fixed 0.7seconds of green time per detected vehicle
YELLOW_TIME = 3.0                   # Duration for yellow transition (seconds)
ALL_RED_TIME = 2.0                  # Duration for all-red clearance phase (seconds)

# YOLOv8 COCO Classes mapping for vehicles
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

LANE_NAMES = ["North", "East", "South", "West"]
NUM_LANES = 4

class IntersectionController:
    def __init__(self):
        self.active_lane_idx = 0  # 0: North, 1: East, 2: South, 3: West
        self.state = "GREEN"      # "GREEN", "YELLOW", "ALL_RED"
        self.state_start_time = 0.0
        
        self.base_green = BASE_GREEN_TIME
        self.max_green = MAX_GREEN_TIME
        self.extra_per_vehicle = TIME_PER_VEHICLE
        self.yellow_time = YELLOW_TIME
        self.all_red_time = ALL_RED_TIME
        
        self.current_allocated_green = self.base_green
        self.consecutive_green_count = 1

    def update(self, current_time, vehicle_counts, emergency_detected_flags):
        elapsed = current_time - self.state_start_time
        
        if self.state == "GREEN":
            # Green time was fixed during transition snapshot. We simply wait for it to expire.
            if elapsed >= self.current_allocated_green:
                self.state = "YELLOW"
                self.state_start_time = current_time
                
        elif self.state == "YELLOW":
            # Transition to all-red clearance after standard yellow duration
            if elapsed >= self.yellow_time:
                self.state = "ALL_RED"
                self.state_start_time = current_time
                
        elif self.state == "ALL_RED":
            if elapsed >= self.all_red_time:
                # Score lanes to pick the next one.
                next_lane_idx = (self.active_lane_idx + 1) % NUM_LANES
                best_score = -1
                
                for i in range(NUM_LANES):
                    idx = (self.active_lane_idx + 1 + i) % NUM_LANES
                    
                    if idx == self.active_lane_idx and self.consecutive_green_count >= 2:
                        continue
                        
                    score = vehicle_counts[idx]
                    if emergency_detected_flags[idx]:
                        score += 999999 # Max priority for emergency
                        
                    if score > best_score:
                        best_score = score
                        next_lane_idx = idx
                        
                if next_lane_idx == self.active_lane_idx:
                    self.consecutive_green_count += 1
                else:
                    self.consecutive_green_count = 1
                
                # Start green phase for the newly selected lane
                self.active_lane_idx = next_lane_idx
                self.state = "GREEN"
                self.state_start_time = current_time
                
                # Snapshot calculation: 7 seconds per vehicle, bounded by min/max limits
                calculated_green = vehicle_counts[next_lane_idx] * self.extra_per_vehicle
                self.current_allocated_green = min(max(self.base_green, calculated_green), self.max_green)
                
    def get_remaining_time(self, current_time):
        elapsed = current_time - self.state_start_time
        if self.state == "GREEN":
            return max(0.0, self.current_allocated_green - elapsed)
        elif self.state == "YELLOW":
            return max(0.0, self.yellow_time - elapsed)
        elif self.state == "ALL_RED":
            return max(0.0, self.all_red_time - elapsed)

    def get_light_state_for_lane(self, lane_idx):
        if lane_idx == self.active_lane_idx:
            if self.state == "GREEN":
                return "GREEN"
            elif self.state == "YELLOW":
                return "YELLOW"
            elif self.state == "ALL_RED":
                return "RED" # Current active lane is in its all-red clearance
        return "RED" # Inactive lanes are always RED


def detect_emergency_vehicle(frame, detections):
    return False


def draw_ui_lane(frame, lane_idx, lane_name, lane_state, vehicle_count, remaining_time, is_active):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    
    # --- 1. Draw Traffic Light Housing (Top Right) ---
    box_w, box_h = 60, 160
    box_x, box_y = w - box_w - 20, 20
    
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (30, 30, 30), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (150, 150, 150), 2)
    
    radius = 20
    center_x = box_x + box_w // 2
    red_y = box_y + 35
    yel_y = box_y + 80
    grn_y = box_y + 125
    
    dim_red = (0, 0, 50)
    dim_yel = (0, 50, 50)
    dim_grn = (0, 50, 0)
    
    bright_red = (0, 0, 255)
    bright_yel = (0, 255, 255)
    bright_grn = (0, 255, 0)
    
    cv2.circle(frame, (center_x, red_y), radius, bright_red if lane_state == 'RED' else dim_red, -1)
    cv2.circle(frame, (center_x, yel_y), radius, bright_yel if lane_state == 'YELLOW' else dim_yel, -1)
    cv2.circle(frame, (center_x, grn_y), radius, bright_grn if lane_state == 'GREEN' else dim_grn, -1)
    
    # --- 2. Draw Text Overlays (Top Left) ---
    y_offset = 50
    
    # Lane Name Highlight
    name_color = (0, 255, 255) if is_active else (255, 255, 255)
    cv2.putText(frame, f"Lane: {lane_name}", (20, y_offset), font, 1.2, name_color, 3, cv2.LINE_AA)
    y_offset += 40
    
    # Vehicle count
    cv2.putText(frame, f"Detected Vehicles: {vehicle_count}", (20, y_offset), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    y_offset += 40
    
    if lane_state == "GREEN":
        color = bright_grn
        text = f"Remaining Green: {remaining_time:.1f}s"
    elif lane_state == "YELLOW":
        color = bright_yel
        text = f"Yellow Time: {remaining_time:.1f}s"
    else: # RED
        color = bright_red
        if lane_state == "RED" and is_active:
             text = f"All-Red Clearance: {remaining_time:.1f}s"
        else:
             text = "Waiting..."
            
    cv2.putText(frame, text, (20, y_offset), font, 1.0, color, 2, cv2.LINE_AA)


def run_traffic_simulation():
    print("Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt') 
    
    # Open video streams for the 4-way intersection
    caps = []
    for i in range(NUM_LANES):
        src = VIDEO_SOURCES[i] if i < len(VIDEO_SOURCES) else VIDEO_SOURCES[0]
        print(f"Opening video source for {LANE_NAMES[i]} Lane: {src}")
        cap = cv2.VideoCapture(src)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video file for Lane {LANE_NAMES[i]}.")
            return
        caps.append(cap)
        
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0 # Fallback fps
    
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize resolution
    out_width = width
    out_height = height
    half_width = width // 2
    half_height = height // 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_processed_4way.mp4', fourcc, fps, (out_width, out_height))
        
    sim_time = 0.0
    dt_per_frame = 1.0 / fps
    
    controller = IntersectionController()
    
    print("Starting processing loop. Press 'q' to exit.")
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frames.append(frame)
            
        if len(frames) < NUM_LANES:
            print("Failed to read frames from sub-streams.")
            break
            
        # 1. Execute Inference in a Batch
        results = model(frames, verbose=False)
        
        vehicle_counts = [0] * NUM_LANES
        emergencies = [False] * NUM_LANES
        
        type_counts = {v: 0 for v in VEHICLE_CLASSES.values()}
        
        # 2. Iterate Detections per Frame
        for i, (frame, result) in enumerate(zip(frames, results)):
            detections = result.boxes
            count = 0
            
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Filter solely for relevant standard vehicle classes
                if cls_id in VEHICLE_CLASSES:
                    count += 1
                    vehicle_type = VEHICLE_CLASSES[cls_id]
                    type_counts[vehicle_type] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box and label
                    label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)  # Box color
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)
            
            vehicle_counts[i] = count
            emergencies[i] = detect_emergency_vehicle(frame, detections)
            
        # 4. State Machine Update
        controller.update(sim_time, vehicle_counts, emergencies)
        remaining_time = controller.get_remaining_time(sim_time)
        
        # 5. UI Overlay Render & Resize
        resized_frames = []
        for i, frame in enumerate(frames):
            lane_state = controller.get_light_state_for_lane(i)
            is_active = (i == controller.active_lane_idx)
            draw_ui_lane(frame, i, LANE_NAMES[i], lane_state, vehicle_counts[i], remaining_time, is_active)
            resized_frames.append(cv2.resize(frame, (half_width, half_height)))
            
        # 6. Stitch frames into a 2x2 grid
        top_row = np.hstack((resized_frames[0], resized_frames[1]))
        bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
        grid = np.vstack((top_row, bottom_row))
        
        # Save Output
        out.write(grid)
        
        analytics_data = {
            "lane_counts": vehicle_counts,
            "type_counts": type_counts,
            "active_lane": LANE_NAMES[controller.active_lane_idx],
            "lane_names": LANE_NAMES,
            "state": controller.state,
            "remaining_time": remaining_time
        }
        
        yield grid, analytics_data
        
        # Increment simulation time mapped exactly to the video frames
        sim_time += dt_per_frame
            
    # Cleanup routines
    for cap in caps:
        cap.release()
    out.release()

def main():
    print("Starting processing loop. Press 'q' to exit.")
    for grid, analytics in run_traffic_simulation():
        cv2.imshow('4-Way Adaptive Traffic Management', grid)
        # Exit strategy: press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
