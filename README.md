# 🚦 AI-Nexus: 4-Way Adaptive Traffic Management System

AI-Nexus is a state-of-the-art, intelligent traffic management system that leverages computer vision and real-time analytics to optimize traffic flow at 4-way intersections. By using YOLOv8 for vehicle detection and a dynamic state-machine controller, the system adjusts green light durations in real-time based on traffic density.

## 🚀 Key Features

- **Real-time Vehicle Detection**: Utilizes YOLOv8 (You Only Look Once) to detect and classify vehicles (Cars, Motorcycles, Buses, Trucks) across four simultaneous video streams.
- **Adaptive Green-Light Timing**: Dynamically calculates green light duration based on the number of vehicles detected in the active lane.
- **Emergency Vehicle Priority**: Integrated logic to prioritize lanes with emergency vehicles (placeholder implemented).
- **Interactive Dashboard**: A sleek Streamlit-based dashboard providing live monitoring, intersection status, and vehicle distribution analytics.
- **Collision Prevention**: Built-in Yellow and All-Red clearance phases to ensure safe transitions between signal changes.
- **Multi-Lane Analytics**: Visual representation of vehicle counts and types per lane through dynamic charts.

## 🛠️ Technology Stack

- **Core Engine**: Python 3.x
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Data Processing**: NumPy, Pandas
- **Frontend Dashboard**: Streamlit

## 📂 Project Structure

- `main.py`: The core simulation engine, including the `IntersectionController` and YOLO detection pipeline.
- `dashboard.py`: The Streamlit application for real-time visualization and monitoring.
- `yolov8n.pt`: Pre-trained YOLOv8 nano model for efficient real-time inference.
- `output_processed_4way.mp4`: Saved output of the processed simulation.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AnkitBharadva/AI-Nexus.git
   cd AI-Nexus
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python numpy ultralytics streamlit pandas
   ```

3. **Configure Video Sources**:
   Update the `VIDEO_SOURCES` list in `main.py` with the paths to your local video files or RTSP streams.

## 🏃 Running the Application

### Option 1: Standard Simulation (OpenCV)
Run the core simulation with a local OpenCV window:
```bash
python main.py
```

### Option 2: Live Analytics Dashboard (Streamlit)
Launch the interactive web-based dashboard:
```bash
streamlit run dashboard.py
```

## 📊 How it Works

1. **Inference**: The system captures frames from four video sources simultaneously.
2. **Detection**: YOLOv8 detects vehicles in each frame, providing real-time counts.
3. **Control Logic**: The `IntersectionController` evaluates the traffic density and decides:
   - Which lane should get the green light next (Prioritizing density and fairness).
   - The optimal duration for the green signal (7 seconds per detected vehicle, within configurable limits).
4. **Visualization**: The processed frames are stitched into a 2x2 grid and displayed with live traffic light overlays and remaining time counters.

---
*Created with ❤️ for smarter and safer cities.*
