# 🤖 Robotics Project – Pick and Place Robot with CoppeliaSim

This project implements a vision-based pick-and-place robot using **CoppeliaSim**, **Machine Vision Toolbox (MVT)**, and **OpenCV**. The robot detects colored shapes in the workspace, classifies them by shape and color, computes real-world positions via homography, and performs precise pick-and-place operations using inverse kinematics.
Tested on CoppaliaSim Edu and a 3-dof real robotic arm

---

## 👤 Team Member

| Student Number | First Name | Last Name |
|----------------|------------|-----------|
| 11427591       | Mohan      | Hao       |

---

## 🧠 Project Overview

### Core Tasks:

1. **Color Blob Detection** – Red, green, and blue blobs are detected using chromaticity thresholds.
2. **Shape Classification** – Blobs are classified into *circle*, *square*, or *other* based on their geometric properties (area & perimeter).
3. **Red Circle Detection for Homography** – Four red calibration circles are detected and used to compute a homography matrix.
4. **Coordinate Transformation** – Blob centroids in image space are converted to real-world coordinates using the homography matrix.
5. **Pick-and-Place Execution** – The robot:
   - Detects and localizes objects
   - Picks them up using a suction gripper
   - Places them at specified target coordinates
   - kinematics and inverse kinematics performance
   - homogeneous and transformation
6. **Evaluation** – After movement, final object positions are compared to the targets, and Euclidean distance errors are calculated.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Main entry point to run the simulation |
| `PickAndPlacerobot.py` | Contains the core pick-and-place logic |
| `coppeliaRobot.py` | Interface for controlling the robot in CoppeliaSim |
| `genericRobotAPI.py` | Generic API for print-only mode (no simulation) |
| `machinevisiontoolbox` | Used for vision processing and image labeling |
| `red_circle_calibration` | Uses red circles to establish workspace homography |

---

## 🔧 Key Functions

### 🔹 Object Detection
- `coloured_objects_blobs()`: Extracts red, green, and blue blobs using chromaticity.
- `calculate_blob_properties()`: Computes area and perimeter of labeled blobs.
- `shape_classification()`: Classifies each blob as circle or square using shape metrics.
- `calculate_blob_centroids()`: Computes centroids of each detected blob.
- `classify_colored_shapes()`: Final color-shape label Project.

### 🔹 Calibration & Homography
- `detect_red_circles()`: Detects fixed-position red circles used for calibration.
- `homography_apply_to_centroids()`: Maps pixel coordinates to world coordinates using the red circles.

### 🔹 Robot Control
- `PickUp()`: Picks up object with a suction cup.
- `Place()`: Places object at the desired location.
- `MoveToHomePosition()`: Moves robot arm away from the camera view.
- `ikine()`: Computes inverse kinematics for the Dobot robot.

### 🔹 Evaluation
- `get_latest_workplace_status()`: Detects current object positions after placing.
- `test_final_positions()`: Calculates placement error in mm per object.

---

## 🧪 How to Run

```bash
python main.py
Use --printonly if you want to test without CoppeliaSim:

bash
Copy
Edit
python main.py --printonly
Make sure CoppeliaSim is running and the appropriate robot scene is loaded.

📌 Notes
Designed for Dobot Magician in the CoppeliaSim environment.

Shape classification thresholds are tuned for the camera angle and simulated lighting conditions.

Red calibration circles must be placed in a fixed order for accurate homography.

📈 Evaluation Metrics
✔️ Correct classification and detection of colored shapes

✔️ Accuracy of object placement (within 5mm error)

✔️ No usage of reference IK or vision functions unless allowed

✔️ Clean modular code following structure provided

🛠 Dependencies
numpy

opencv-python

machinevisiontoolbox

matplotlib

coppeliaSim (running instance)

CoppeliaSim remote API installed in Python

✅ Completed
 Color-based blob detection

 Area/perimeter-based shape classification

 Homography from red calibration markers

 Inverse kinematics

 Pick-and-place execution

 Final position validation

📬 Contact
If you encounter any issues running the simulation, please contact:

📧 imhaom@gmail.com
