# Face Attendance System

Simple face-based attendance using webcam capture and CSV logging.

## Features
- Register a user by capturing face images from webcam
- Recognize faces live and mark attendance in CSV
- Stores registered face photos locally

## Requirements
- Python 3.9+
- Webcam

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
Register a user:
```bash
python main.py register --name "Alice"
```

Start attendance:
```bash
python main.py attend
```

## Web App
Install requirements, then run:
```bash
python app.py
```

Open in browser: http://127.0.0.1:5000

Default teacher login:
- Username: teacher
- Password: teacher123

Default student login:
- Username: student
- Password: student123

Teacher can add students, upload face photos, and view/download attendance.
Students can log in to view their own attendance.

## Files
- `data/known_faces/` - stored user face photos
- `attendance/attendance.csv` - attendance log

## Notes
- This uses OpenCV's LBPH recognizer (no `dlib` compile step).
- Tweak the `--threshold` value if recognition is too strict or too loose.
