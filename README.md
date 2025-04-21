# Key Tracking System with YOLOv8

This project uses YOLOv8 object detection to track and monitor the location of personal keys (e.g., house key, work key) in a workspace or home environment using AI-powered camera input.

## 📁 Project Structure
- `video_keys.mp4` – Input video used for key detection
- `output_with_boxes.mp4` – YOLO-generated video output with bounding boxes
- `train/` – Folder containing training images and annotations
- `docs/` – Project documentation and reports
- `notebooks/` – Jupyter/Colab notebooks for training and inference
- `scripts/` – Python scripts used for detection and video processing

## 🚀 Features
- Real-time key tracking using YOLOv8
- Movement detection and notification
- Console alert: `"The 'key_name' has been moved"` after stable movement
- Optimized for limited hardware (Colab/Local)

## 🛠 Technologies Used
- Python
- OpenCV
- Ultralytics YOLOv8
- Google Colab / Jupyter Notebooks

## 📄 How to Use
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
