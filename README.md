# 🦺 Personal Protective Equipment System

> An AI-powered PPE detection and monitoring system designed to identify personal protective equipment in images, videos, and real-time camera feeds.

## 📌 Overview

Personal Protective Equipment System is an AI-based computer vision application designed to monitor whether required safety equipment is being properly used.

The system can analyze images, videos, and real-time camera feeds to detect different types of personal protective equipment and identify potential safety violations.

The application provides a simple monitoring interface and can maintain violation records for further analysis.

## ✨ Features

- 🦺 PPE detection using AI
- 📷 Real-time camera detection
- 🖼️ Image-based PPE detection
- 🎥 Video-based PPE detection
- 🪖 Helmet detection
- 😷 Mask detection
- 🦺 Safety vest detection
- 🧤 Safety glove detection
- 🥾 Safety boot detection
- ⚠️ Safety violation detection
- 📊 Violation monitoring and reporting
- 📝 Violation log management
- 📁 Image and video analysis

## 🧰 Technologies Used

- Python
- Streamlit
- YOLO / Ultralytics
- OpenCV
- Pandas
- Pillow
- Computer Vision
- Machine Learning

## 🏗️ System Architecture

```text
Camera / Image / Video
          ↓
      Input Data
          ↓
     YOLO Detection
          ↓
   PPE Identification
          ↓
 ┌──────────────────────┐
 │ Helmet               │
 │ Mask                 │
 │ Safety Vest          │
 │ Gloves               │
 │ Safety Boots         │
 └──────────────────────┘
          ↓



## 👨‍💻 Author

**Md. Aminul Haque Khan**

- 🎓 Computer Science & Engineering Student
- 🤖 AI & Computer Vision Enthusiast
- 💻 Software & Full-Stack Developer
- 🛠️ Interested in AI, Machine Learning, Web Development & Software Engineering
- 🔗 GitHub: [@mdaminulhaquekhan](https://github.com/mdaminulhaquekhan)
   Violation Detection
          ↓
    Monitoring / Logs
