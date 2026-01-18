# 🖐️ Computer Vision Keyboard  
### A Touchless Gesture-Based Virtual Keyboard

> A modern, touchless text input system that enables typing using hand gestures and computer vision.

---

## 🚀 Overview

**Computer Vision Keyboard** is a gesture-controlled virtual keyboard that allows users to type text **without touching a physical keyboard**.  
The system uses a webcam to track hand movements in real time and converts them into text input using computer vision techniques.

Characters are arranged in a **circular rotary keyboard layout**, where users rotate their finger to navigate through characters and use **pinch gestures** to select them.  
A lightweight **predictive text system** is integrated to improve typing speed and user experience.

This project demonstrates an innovative approach to **human–computer interaction (HCI)** using vision-based gesture recognition.

---

## ✨ Key Features

- 🖐️ Touchless text input using hand gestures  
- 🔄 Circular rotary keyboard layout  
- 🤏 Gesture-based selection:
  - Short pinch → select character
  - Long pinch → select predicted word
- 🧠 Top-3 predictive text suggestions  
- 🖥️ Types directly into any active application  
- ⚡ Real-time performance  
- ❌ No calibration required  

---

## 🛠️ Technologies Used

- **Python 3.11**
- **OpenCV** – real-time webcam video processing
- **MediaPipe Hands** – hand landmark detection
- **NumPy** – mathematical computations
- **PyAutoGUI** – system-level keyboard input simulation

---

## 📐 Interaction Design

| Gesture | Action |
|------|------|
| Rotate index finger | Navigate characters |
| Short pinch | Select letter / SPACE / BACKSPACE |
| Long pinch | Select predicted word |
| ESC key | Exit application |

To avoid gesture conflicts, the system uses **time-based gesture differentiation**.

---

## 🧠 Predictive Text System

- Dictionary-based prediction (offline)
- Displays the **top 3 word suggestions**
- Suggestions update dynamically as the user types
- Selecting a prediction:
  - Deletes the typed prefix
  - Inserts the full predicted word correctly
- Lightweight and fast (no machine learning models)

---

## 🧪 How to Run the Project

```bash
# Step 1: Clone the repository
git clone https://github.com/Charan-516/computer-vision-keyboard.git
cd computer-vision-keyboard

# Step 2: Create a virtual environment
python -m venv venv

# Step 3: Activate the virtual environment
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Run the application
python gesture_keyboard_prediction_final.py
